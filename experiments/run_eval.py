"""Correct isolated partial-feedback bandit training CLI.

Generate-all is intentionally absent from training. Each episode selects one
arm, performs one generation, observes one reward, and updates one arm.
"""
from __future__ import annotations

import argparse, hashlib, importlib.metadata, json, platform, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config import *
from llm.generate import LLMClient, GenerationError
from rag.retrieve import retrieve
from rl.bandit import EpsilonGreedyBandit
from scoring.reward import citation_metrics, semantic_support_score_v2
from utils.io import read_jsonl


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


EXPECTED_INPUT_HASHES = {
    "questions": "5312a60374a2d72754b29eb8fba4f73f90b99178aa0d56b16694f133a7f62e99",
    "chunks": "01cbbf11efc906d7ccc5b06a96f8041a7e7c148a130ebe39087a93edfe636564",
    "index": "0b1aabe27f683036443ac9f130f6ac79f6c61a57bd4ce099eeb1c4523d58c971",
}


def validate_final_inputs():
    expected={
        DATA_DIR/'questions.jsonl':EXPECTED_INPUT_HASHES['questions'],
        CHUNKS_JSONL:EXPECTED_INPUT_HASHES['chunks'],
        FAISS_INDEX:EXPECTED_INPUT_HASHES['index']}
    mismatches={}
    for p, expected_hash in expected.items():
        actual = sha256_file(p) if p.exists() else None
        if actual != expected_hash:
            mismatches[str(p)] = (actual, expected_hash)
    if mismatches: raise RuntimeError(f'Final input integrity failure: {mismatches}')
    return {str(p): expected_hash for p, expected_hash in expected.items()}


def _package_versions():
    versions={}
    for package in ['google-genai','faiss-cpu','sentence-transformers','numpy','pandas']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    return versions


def _optional_sha256(path: Path):
    return sha256_file(path) if path.exists() else None


def run_training_loop(num_episodes=TRAIN_EPISODES, seed=MASTER_SEED, output_root=None,
                      client=None, retriever=retrieve, scorer=semantic_support_score_v2,
                      bandit_factory=None, validate_inputs=True):
    """Run a fresh isolated partial-feedback experiment.

    Dependency injection is deliberate: regression tests can prove call order,
    single-generation feedback, and failure behavior without making API calls.
    """
    if validate_inputs:
        validate_final_inputs()
    random.seed(seed)
    questions=read_jsonl(DATA_DIR/'questions.jsonl')
    train_ids=['q042','q047','q048','q016','q010','q017','q025','q035','q032','q001','q045','q028','q034','q006','q030','q012','q037','q002','q022','q003','q044','q036','q024','q041','q011','q023','q019','q050','q021','q008','q043','q015','q029','q039']
    by_id={q['id']:q for q in questions}; train=[by_id[x] for x in train_ids]
    run_id=f"corrected_seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    root=Path(output_root or OUTPUTS_DIR/'final_revision_nonlatex'/'corrected_runs')
    run_dir=root/run_id; run_dir.mkdir(parents=True, exist_ok=False)
    client=client or LLMClient(provider='gemini', final_mode=True)
    if client.provider!='gemini': raise RuntimeError('Final runs require Gemini; dry-run fallback is forbidden.')
    bandit=(bandit_factory or EpsilonGreedyBandit)(ACTIONS, DEFAULT_EPSILON, EPSILON_DECAY, MIN_EPSILON)
    rows=[]; snapshots=[]
    metadata={"run_id":run_id,"created_at_utc":datetime.now(timezone.utc).isoformat(),"seed":seed,"split":"train","episodes":num_episodes,
              "llm_provider":"gemini","llm_model":DEFAULT_GEMINI_MODEL,"embedding_model":DEFAULT_EMBED_MODEL,
              "semantic_similarity_threshold":DEFAULT_SENT_SIM_THRESHOLD,"reward_mode":"semantic_proxy","semantic_scorer_version":SEMANTIC_SCORER_CLEAN_V2,
              "nli_model":NLI_MODEL_NAME,"nli_enabled_for_evaluation":False,"top_k":DEFAULT_TOP_K,"chunk_size_words":CHUNK_SIZE_WORDS,
              "chunk_overlap_words":CHUNK_OVERLAP_WORDS,"initial_epsilon":DEFAULT_EPSILON,"epsilon_decay":EPSILON_DECAY,"min_epsilon":MIN_EPSILON,
              "final_experiment_mode":True,"controlled_temperature":None,"action_temperatures":ACTION_TEMPERATURES,
              "question_file_sha256":sha256_file(DATA_DIR/'questions.jsonl'),"chunks_file_sha256":_optional_sha256(CHUNKS_JSONL),"index_file_sha256":_optional_sha256(FAISS_INDEX),
              "prompt_configuration_sha256":_optional_sha256(REPO_ROOT/'llm'/'prompts.py'),
              "source_notebook_sha256":_optional_sha256(REPO_ROOT/'REVISED_ALL_IN_ONE_hallucination_penalty.ipynb'),
              "code_hashes":{"run_eval.py":sha256_file(Path(__file__)),"reward.py":sha256_file(REPO_ROOT/'scoring'/'reward.py')},
              "python_version":platform.python_version(),"package_versions":_package_versions(),"runtime":platform.platform()}
    (run_dir/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    try:
        for episode in range(num_episodes):
            q=train[episode%len(train)]
            contexts,sources,hits=retriever(q['question'], top_k=DEFAULT_TOP_K, require_nontrivial_corpus=True)
            action=bandit.choose_action()
            started=time.perf_counter()
            generation=client.generate(action,q['question'],hits,temperature=ACTION_TEMPERATURES[action],citation_enforced=False)
            latency=time.perf_counter()-started
            answer=generation.answer if hasattr(generation,'answer') else str(generation)
            score=scorer(answer,'\n'.join(contexts))
            cites=citation_metrics(answer,sources)
            bandit.update(action,float(score['reward'])); bandit.validate_epsilon()
            context_hash=hashlib.sha256('\n'.join(contexts).encode()).hexdigest()
            rows.append({"run_id":run_id,"episode":episode,"question_id":q['id'],"question":q['question'],"chosen_action":action,
                         "temperature":ACTION_TEMPERATURES[action],"chosen_reward":score['reward'],"groundedness":score['groundedness'],
                         "penalty":score['hallucination_penalty'],"supported":score['supported'],"unsupported":score['unsupported'],
                         "is_abstention":score['is_abstention'],"epsilon_after":bandit.epsilon,"bandit_updates":bandit.total_updates,
                         "citation_count":cites['citation_count'],"citation_precision":cites['citation_precision'],
                         "citation_sentence_coverage":cites['citation_sentence_coverage'],"latency_seconds":latency,
                         "llm_provider":client.provider,"llm_model":DEFAULT_GEMINI_MODEL,
                         "sources":json.dumps(sources),"context_hash":context_hash,"scorer_version":SEMANTIC_SCORER_CLEAN_V2,"answer":answer})
            snapshots.append({"run_id":run_id,"episode":episode,"ordered_source_ids":sources,"retrieved_chunk_texts":contexts,
                              "retrieval_similarity_scores":[h.get('score') for h in hits],"context_sha256":context_hash})
            pd.DataFrame(rows).to_csv(run_dir/'episodes.csv',index=False)
            (run_dir/'context_snapshots.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in snapshots)+'\n',encoding='utf-8')
            bandit.save(run_dir/'bandit_state.json')
    except Exception as exc:
        (run_dir/'RUN_FAILED.json').write_text(json.dumps({"run_id":run_id,"failed_episode":len(rows),"error_type":type(exc).__name__,"error":str(exc)},indent=2))
        raise
    (run_dir/'RUN_COMPLETE.json').write_text(json.dumps({"run_id":run_id,"rows":len(rows),"total_updates":bandit.total_updates,"final_epsilon":bandit.epsilon},indent=2))
    return run_dir


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--episodes',type=int,default=TRAIN_EPISODES); p.add_argument('--seed',type=int,default=MASTER_SEED)
    args=p.parse_args(); print(run_training_loop(args.episodes,args.seed))
