# Reproducibility

## Setup

Use Python 3.12 where possible:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `GOOGLE_API_KEY` in the environment or an ignored local `.env`; never commit it. For another checkout location, run from the repository root or set `PROJECT_ROOT` before importing `config.py`.

## Corpus and index

The third-party text-bearing index is not redistributed. Rebuild the 98-document corpus with the authoritative notebook's Wikipedia API cells and then run:

```bash
python rag/build_index.py
python scripts/inspect_index.py
```

Expected output is 1,245 chunks embedded by `sentence-transformers/all-MiniLM-L6-v2` in a FAISS `IndexFlatIP`. Compare hashes with `artifacts/hashes/INPUT_INTEGRITY_REPORT.json`.

## Tests and audit

```bash
python -m unittest discover -s tests -v
python experiments/final_nonlatex_workflow.py --audit-only
```

## Experiments

Each generation command sends questions and retrieved chunks to Gemini and can incur API cost:

```bash
python experiments/final_nonlatex_workflow.py --rescore-historical
python experiments/final_nonlatex_workflow.py --run-primary
python experiments/final_nonlatex_workflow.py --run-multiseed
python experiments/final_nonlatex_workflow.py --run-test-baselines
python experiments/final_nonlatex_workflow.py --run-validation-selection
python experiments/final_nonlatex_workflow.py --run-controlled-temperature
python experiments/final_nonlatex_workflow.py --run-nli
python experiments/final_nonlatex_workflow.py --run-threshold-sensitivity
python experiments/final_nonlatex_workflow.py --run-retrieval-diagnostics
python experiments/final_nonlatex_workflow.py --run-statistics
```

`--all` executes the full workflow. Generation failures are not scored and do not update the bandit. Fresh runs use isolated state. The permanent split manifests are under `data/`.

The automatic semantic reward and BART-MNLI measure retrieved-context support, not factual truth. Historical runs and post-hoc rescoring must remain separate from fresh corrected online experiments.



## Public-release context sanitization

The 95-document third-party corpus and text-bearing processed artifacts are not redistributed. Public experiment artifacts retain generated responses, source IDs, retrieval order/similarities where recorded, context hashes, scores, and metrics; full retrieved-context text has been removed. These artifacts support result auditability, while full retrieval/LLM re-execution requires reconstructing the corpus and FAISS index.

