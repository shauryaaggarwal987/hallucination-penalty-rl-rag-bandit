# Reward-Driven Hallucination Control in Retrieval-Augmented Generation via Bandit Learning

## Overview

This research prototype combines dense retrieval with five Gemini generation configurations (A0-A4) and a non-contextual epsilon-greedy bandit. Each training episode selects one configuration, makes exactly one generation call, scores only that response with an automatic semantic-support reward, and updates only the selected arm. It studies global strategy preference under partial feedback; it does not learn contextual per-query routing or fine-tune model weights.

Results should be interpreted cautiously. Semantic support and automatic NLI measure consistency with retrieved context, not independently verified factual correctness.

## Published Research Paper

**Title:** Reward-driven hallucination control in retrieval-augmented generation via bandit learning

**Authors:** Shaurya Aggarwal, Ishani Lohar, Gireesh Kumar, Richa Sharma

**Journal:** EPJ Web Conf.

**Publisher:** EDP Sciences

**Volume:** 391 (2026)

**Article Number:** 01004

**Published Online:** October 15, 2026

**Conference:** International Conference on Artificial Intelligence for Sustainable Development (AI4SD 2026)

**DOI:** [10.1051/epjconf/202639101004](https://doi.org/10.1051/epjconf/202639101004)

**Publication:** [Read the published paper](https://doi.org/10.1051/epjconf/202639101004)

## Repository Contents

- `notebooks/`: authoritative experimental notebook and sanitization report.
- `rag/`, `llm/`, `scoring/`, `rl/`, `experiments/`: active implementation.
- `data/`: frozen questions, split, sample documents, and corpus metadata.
- `artifacts/`: frozen input hashes and index reconstruction notes.
- `results/`: clearly separated historical, historical-rescored, and corrected experiments.
- `figures/`: final traceable scientific plots.
- `reports/` and `docs/`: reproducibility, scorer, provenance, and handoff records.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key"
python -m unittest discover -s tests -v
python experiments/final_nonlatex_workflow.py --audit-only
```

Corpus/index reconstruction is required before retrieval or Gemini experiments; see `DATASET_AND_CORPUS.md`.

## Reproducing the Experiments

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The workflow implements historical rescoring, primary/multi-seed training, test baselines, validation selection, controlled-temperature analysis, NLI, threshold sensitivity, retrieval diagnostics, statistics, and `--all`.

## Data and Corpus

The question bank contains 50 questions with a fixed 34/7/9 split. Experiments used 98 documents and 1,245 overlapping chunks. Third-party raw text is omitted pending complete per-revision attribution; metadata and reconstruction instructions are included.

## Historical vs Corrected Results

- `results/historical/`: frozen runs made with the historical scorer.
- `results/historical/rescored/`: post-hoc clean-v2 rescoring of fixed historical responses.
- `results/corrected/`: newly generated corrected online runs and evaluations, each included only with completion evidence where applicable.

Historical rescoring is not corrected online training.

## Automatic Metrics

The primary reward is a sentence-embedding semantic-support proxy. BART-MNLI is a separate context-entailment proxy. Neither constitutes factual ground truth; abstention, answer rate, citation behavior, and unsupported-unit rates should be reported alongside them.

## Results

The corrected seed-42 run learned A2 as its final greedy action and had mean training reward 0.5561. On nine held-out questions, A2 and the validation-selected fixed policy had mean reward 0.6667; the generate-all score-max comparator had 0.7265 but requires five calls per question. Across 430 substantive scoring units, semantic/NLI raw agreement was 0.7163 and Cohen's kappa was 0.3559. See `RESULTS.md` for provenance and limitations.

## Citation

If you use this research or its implementation, please cite the published paper.

**Official publisher-provided BibTeX:**

```bibtex
@article{ refId0,
	author = {{Aggarwal, Shaurya} and {Lohar, Ishani} and {Kumar, Gireesh} and {Sharma, Richa}},
	title = {Reward-driven hallucination control in retrieval-augmented generation via bandit learning},
	DOI= "10.1051/epjconf/202639101004",
	url= "https://doi.org/10.1051/epjconf/202639101004",
	journal = {EPJ Web Conf.},
	year = 2026,
	volume = 391,
	pages = "01004",
}
```

**Paper DOI:** https://doi.org/10.1051/epjconf/202639101004

For machine-readable citation metadata, see [CITATION.cff](CITATION.cff).
```

For machine-readable citation metadata, see [CITATION.cff](CITATION.cff).
## License

No explicit software license has been selected. See [LICENSE_STATUS.md](LICENSE_STATUS.md) and [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).



## Public-release context sanitization

The 95-document third-party corpus and text-bearing processed artifacts are not redistributed. Public experiment artifacts retain generated responses, source IDs, retrieval order/similarities where recorded, context hashes, scores, and metrics; full retrieved-context text has been removed. These artifacts support result auditability, while full retrieval/LLM re-execution requires reconstructing the corpus and FAISS index.

