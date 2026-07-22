"""Versioned semantic-support scoring and citation evaluation.

The legacy scorer is retained for frozen-run reproducibility. New scientific
runs must use ``semantic_proxy_clean_v2``. Generic document chunking is not
modified here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config import (
    ABSTENTION_TEXT,
    DEFAULT_EMBED_MODEL,
    DEFAULT_SENT_SIM_THRESHOLD,
    SEMANTIC_SCORER_CLEAN_V2,
    SEMANTIC_SCORER_LEGACY_V1,
)
from utils.text import sentence_split

CITATION_RE = re.compile(r"\[source:\s*([^\]]+?)\]", re.IGNORECASE)
LIST_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[-*+\u2022])\s*")
MARKDOWN_EMPHASIS_RE = re.compile(r"(?:\*\*|__|(?<!\*)\*(?!\*)|(?<!_)_(?!_))")
_SCORING_MODEL = None


def semantic_sentence_split(text: str) -> List[str]:
    """Historical notebook sentence splitter with a dependency-safe fallback."""
    value=(text or "").strip()
    if not value:
        return []
    try:
        import nltk
        return [x.strip() for x in nltk.sent_tokenize(value) if x.strip()]
    except Exception:
        return sentence_split(value)


@dataclass(frozen=True)
class ScoringUnit:
    raw_unit: str
    cleaned_unit: str
    citations: tuple[str, ...]


def is_strict_abstention(answer: str) -> bool:
    cleaned = re.sub(r"[\s\.,;:!?]+$", "", (answer or "").strip())
    return cleaned.casefold() == ABSTENTION_TEXT.casefold()


def extract_citations(text: str) -> List[str]:
    """Return individual source IDs from citation tags.

    A model may emit a compact tag such as ``[source: a#0, b#2]``.  The tag
    still represents two citations, so split each matched payload on commas
    while retaining the raw tag in :class:`ScoringUnit.raw_unit`.
    """
    citations: List[str] = []
    for payload in CITATION_RE.findall(text or ""):
        citations.extend(part.strip() for part in payload.split(",") if part.strip())
    return citations


def strip_citations_for_embedding(text: str) -> str:
    return re.sub(r"\s+", " ", CITATION_RE.sub(" ", text or "")).strip()


def is_pure_formatting_unit(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    value = CITATION_RE.sub("", value).strip()
    value = MARKDOWN_EMPHASIS_RE.sub("", value).strip()
    if not value:
        return True
    if re.fullmatch(r"\d+[.)]?", value):
        return True
    if re.fullmatch(r"[-*+\u2022]+", value):
        return True
    return re.search(r"[\w]", value, flags=re.UNICODE) is None


def _raw_response_units(answer_text: str) -> List[str]:
    """Split response text while preserving list-marker/citation association."""
    units: List[str] = []
    for line in re.split(r"[\r\n]+", answer_text or ""):
        line = line.strip()
        if not line:
            continue
        # A substantive list item remains one unit; ordinary prose can use the
        # historical sentence splitter without affecting document chunking.
        if LIST_PREFIX_RE.match(line) and not is_pure_formatting_unit(line):
            units.append(line)
        else:
            # Source IDs frequently contain periods.  Protect complete citation
            # tags while sentence-tokenizing so ``doc.txt#0`` cannot be split
            # into fragments and lose its association with the response unit.
            protected=[]
            def stash(match):
                protected.append(match.group(0))
                return f" CITATIONTOKEN{len(protected)-1} "
            tokenized=CITATION_RE.sub(stash,line)
            split=semantic_sentence_split(tokenized) or [tokenized]
            for value in split:
                for i,citation in enumerate(protected):
                    value=value.replace(f"CITATIONTOKEN{i}",citation)
                units.append(value.strip())
    return units


def extract_scoring_units(answer_text: str) -> List[ScoringUnit]:
    if is_strict_abstention(answer_text):
        return []
    result: List[ScoringUnit] = []
    for raw in _raw_response_units(answer_text):
        if is_pure_formatting_unit(raw):
            continue
        cleaned = LIST_PREFIX_RE.sub("", raw, count=1)
        cleaned = strip_citations_for_embedding(cleaned)
        cleaned = MARKDOWN_EMPHASIS_RE.sub("", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n")
        if is_pure_formatting_unit(cleaned):
            continue
        result.append(ScoringUnit(raw, cleaned, tuple(extract_citations(raw))))
    return result


def get_scoring_model(model_name: str = DEFAULT_EMBED_MODEL):
    global _SCORING_MODEL
    if _SCORING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SCORING_MODEL = SentenceTransformer(model_name)
    return _SCORING_MODEL


def _empty_result(version: str, *, abstention: bool, reward: float) -> Dict[str, Any]:
    return {
        "scorer_version": version, "is_abstention": abstention,
        "n_sentences": 0, "supported": 0, "unsupported": 0,
        "supported_fraction": 0.0, "groundedness": 0.0,
        "hallucination_penalty": 0.0 if abstention else 1.0,
        "reward": reward, "per_sentence": [], "formatting_units_removed": 0,
        "citations_removed_before_embedding": 0,
    }


def _score_units(units: Sequence[ScoringUnit], context_sentences: Sequence[str],
                 threshold: float, version: str, model=None) -> Dict[str, Any]:
    if not units:
        return _empty_result(version, abstention=False, reward=-1.0)
    if not context_sentences:
        details = [{**asdict(u), "max_similarity": 0.0, "supported": False} for u in units]
    else:
        model = model or get_scoring_model()
        context_embeddings = np.asarray(model.encode(list(context_sentences), normalize_embeddings=True), dtype="float32")
        answer_embeddings = np.asarray(model.encode([u.cleaned_unit for u in units], normalize_embeddings=True), dtype="float32")
        similarities = answer_embeddings @ context_embeddings.T
        details = []
        for i, unit in enumerate(units):
            sim = float(similarities[i].max())
            details.append({**asdict(unit), "max_similarity": sim, "supported": sim >= threshold})
    supported = sum(bool(d["supported"]) for d in details)
    n = len(details)
    fraction = supported / n
    return {
        "scorer_version": version, "is_abstention": False,
        "n_sentences": n, "supported": supported, "unsupported": n-supported,
        "supported_fraction": fraction, "groundedness": fraction,
        "hallucination_penalty": (n-supported)/n, "reward": 2*fraction-1,
        "per_sentence": details,
        "formatting_units_removed": 0,
        "citations_removed_before_embedding": sum(bool(u.citations) for u in units),
    }


def semantic_support_score_v2(answer_text: str, retrieved_context_text: str,
                              sim_threshold: float = DEFAULT_SENT_SIM_THRESHOLD,
                              reward_mode: str = "semantic_proxy",
                              answerable: Optional[bool] = None, model=None) -> Dict[str, Any]:
    if is_strict_abstention(answer_text):
        reward = (-1.0 if answerable else 1.0) if reward_mode == "answerability_aware" and answerable is not None else 0.0
        return _empty_result(SEMANTIC_SCORER_CLEAN_V2, abstention=True, reward=reward)
    raw_units = _raw_response_units(answer_text)
    units = extract_scoring_units(answer_text)
    result = _score_units(units, semantic_sentence_split(retrieved_context_text or ""), sim_threshold, SEMANTIC_SCORER_CLEAN_V2, model)
    result["formatting_units_removed"] = max(0, len(raw_units)-len(units))
    return result


def semantic_support_score_legacy_v1(answer_text: str, retrieved_context_text: str,
                                     sim_threshold: float = DEFAULT_SENT_SIM_THRESHOLD,
                                     model=None) -> Dict[str, Any]:
    if is_strict_abstention(answer_text):
        return _empty_result(SEMANTIC_SCORER_LEGACY_V1, abstention=True, reward=0.0)
    legacy = [ScoringUnit(s, s, tuple(extract_citations(s))) for s in semantic_sentence_split(answer_text or "")]
    return _score_units(legacy, semantic_sentence_split(retrieved_context_text or ""), sim_threshold, SEMANTIC_SCORER_LEGACY_V1, model)


def score_answer(answer_text: str, retrieved_context_text: str,
                 sim_threshold: float = DEFAULT_SENT_SIM_THRESHOLD, **kwargs) -> Dict[str, Any]:
    """Compatibility entrypoint; corrected v2 is the new default."""
    return semantic_support_score_v2(answer_text, retrieved_context_text, sim_threshold, **kwargs)


def citation_metrics(answer: str, valid_source_ids: Sequence[str]) -> Dict[str, Any]:
    citations = extract_citations(answer)
    valid_set = set(valid_source_ids)
    valid = [c for c in citations if c in valid_set]
    invalid = [c for c in citations if c not in valid_set]
    units = extract_scoring_units(answer)
    covered = sum(any(c in valid_set for c in u.citations) for u in units)
    coverage = covered/len(units) if units else (1.0 if is_strict_abstention(answer) else 0.0)
    return {
        "citation_count": len(citations), "valid_citation_count": len(valid),
        "invalid_citation_count": len(invalid),
        "citation_precision": len(valid)/len(citations) if citations else 0.0,
        "citation_sentence_coverage": coverage, "citations": citations,
        "invalid_citations": invalid, "substantive_unit_count": len(units),
    }
