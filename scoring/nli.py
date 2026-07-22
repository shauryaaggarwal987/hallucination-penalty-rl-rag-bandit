"""Context-conditioned NLI evaluation (evaluation only; never a reward)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import numpy as np

from config import NLI_MODEL_NAME, NLI_SUPPORT_THRESHOLD
from scoring.reward import extract_scoring_units, is_strict_abstention

_TOKENIZER = None
_MODEL = None


def _label_index(config, needle: str) -> int:
    labels = {int(k): str(v).lower() for k, v in config.id2label.items()}
    for idx, label in labels.items():
        if needle in label:
            return idx
    reverse = {str(k).lower(): int(v) for k, v in getattr(config, "label2id", {}).items()}
    for label, idx in reverse.items():
        if needle in label:
            return idx
    raise ValueError(f"Cannot find {needle!r} label in NLI model config")


def get_nli_resources(model_name: str = NLI_MODEL_NAME):
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
        _MODEL.eval()
    return _TOKENIZER, _MODEL


def score_premise_hypothesis(premise: str, hypothesis: str, *, tokenizer=None, model=None) -> Dict[str, float]:
    if tokenizer is None or model is None:
        tokenizer, model = get_nli_resources()
    encoded = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**encoded)
    logits = outputs.logits.detach().cpu().numpy()[0]
    probs = np.exp(logits - logits.max()); probs = probs / probs.sum()
    config = model.config
    return {
        "entailment_probability": float(probs[_label_index(config, "entail")]),
        "contradiction_probability": float(probs[_label_index(config, "contrad")]),
        "neutral_probability": float(probs[_label_index(config, "neutral")]),
    }


def score_premise_hypothesis_batch(premises, hypothesis, *, tokenizer=None, model=None):
    """Score one hypothesis against several retrieved chunks in one batch."""
    if tokenizer is None or model is None:
        tokenizer, model = get_nli_resources()
    encoded = tokenizer(list(premises), [hypothesis] * len(premises), return_tensors='pt',
                        padding=True, truncation=True, max_length=512)
    outputs = model(**encoded)
    logits = outputs.logits.detach().cpu().numpy()
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    config=model.config; ent=_label_index(config,'entail'); con=_label_index(config,'contrad'); neu=_label_index(config,'neutral')
    return [{'entailment_probability':float(p[ent]),'contradiction_probability':float(p[con]),
             'neutral_probability':float(p[neu])} for p in probs]


def nli_support_score(answer_text: str, retrieved_chunks: Sequence[str],
                      source_ids: Optional[Sequence[str]] = None,
                      threshold: float = NLI_SUPPORT_THRESHOLD,
                      tokenizer=None, model=None) -> Dict[str, Any]:
    if is_strict_abstention(answer_text):
        return {"enabled": True, "is_abstention": True, "nli_supported_fraction": np.nan, "per_sentence": []}
    units = extract_scoring_units(answer_text)
    source_ids = list(source_ids or [str(i) for i in range(len(retrieved_chunks))])
    details = []
    for unit in units:
        comparisons = score_premise_hypothesis_batch(retrieved_chunks, unit.cleaned_unit,
                                                      tokenizer=tokenizer, model=model) if retrieved_chunks else []
        if comparisons:
            best_i = max(range(len(comparisons)), key=lambda i: comparisons[i]["entailment_probability"])
            best = comparisons[best_i]
            max_contradiction = max(x["contradiction_probability"] for x in comparisons)
            supported = best["entailment_probability"] >= threshold and best["entailment_probability"] == max(best.values())
            details.append({
                "raw_unit": unit.raw_unit, "cleaned_unit": unit.cleaned_unit,
                "max_entailment_probability": best["entailment_probability"],
                "max_contradiction_probability": max_contradiction,
                "best_entailing_source": source_ids[best_i], "nli_supported": bool(supported),
            })
        else:
            details.append({"raw_unit": unit.raw_unit, "cleaned_unit": unit.cleaned_unit,
                            "max_entailment_probability": 0.0, "max_contradiction_probability": 0.0,
                            "best_entailing_source": None, "nli_supported": False})
    fraction = float(np.mean([x["nli_supported"] for x in details])) if details else np.nan
    return {"enabled": True, "is_abstention": False, "nli_supported_fraction": fraction, "per_sentence": details}
