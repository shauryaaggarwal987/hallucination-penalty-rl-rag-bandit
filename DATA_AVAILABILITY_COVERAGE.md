# Data-Availability Coverage

| Paper data-availability item | Repository location | Included? | Notes |
|---|---|---:|---|
| Source code | `app/`, `rag/`, `llm/`, `scoring/`, `rl/`, `experiments/`, `utils/`, `scripts/` | Yes | Active corrected implementation. |
| Experimental notebook | `notebooks/REVISED_ALL_IN_ONE_hallucination_penalty.ipynb` | Yes | Public-safety reviewed copy. |
| Question bank | `data/questions.jsonl` | Yes | 50 frozen questions. |
| Split | `data/split_manifest.*` | Yes | 34/7/9, no overlap. |
| Corpus metadata | `data/corpus_manifest.csv`, `DATASET_AND_CORPUS.md` | Yes | Raw external text excluded. |
| Processed index | `artifacts/index/` | No | Hashes and reconstruction instructions included; text-bearing artifacts excluded for provenance/licensing caution. |
| Historical runs | `results/historical/` | Yes | Historical scorer clearly labeled. |
| Corrected runs | `results/corrected/` | Yes | Complete runs and final analyses. |
| Figures | `figures/` | Yes | Final scientific figures with manifest. |
| Reproducibility report | `REPRODUCIBILITY.md`, `reports/` | Yes | Clean-checkout guidance and final audit reports. |



## Public-release context sanitization

The 95-document third-party corpus and text-bearing processed artifacts are not redistributed. Public experiment artifacts retain generated responses, source IDs, retrieval order/similarities where recorded, context hashes, scores, and metrics; full retrieved-context text has been removed. These artifacts support result auditability, while full retrieval/LLM re-execution requires reconstructing the corpus and FAISS index.

