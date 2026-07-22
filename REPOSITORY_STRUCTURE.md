# Repository Structure

```text
.
├── notebooks/       authoritative notebook and public-safety review
├── app/             optional interactive application
├── rag/             ingestion, indexing, and retrieval
├── llm/             prompts and Gemini generation adapter
├── scoring/         semantic reward and NLI proxy
├── rl/              epsilon-greedy bandit
├── experiments/     training, evaluation, plots, final workflow
├── utils/           shared I/O, environment, and text helpers
├── scripts/         smoke tests and index inspection
├── tests/           regression suite
├── data/            questions, split, samples, corpus metadata
├── artifacts/       frozen hashes and index reconstruction notes
├── results/         historical and corrected scientific artifacts
├── figures/         selected final figures and manifest
├── reports/         final technical reports
└── docs/            changelogs, legacy map, and project summary
```

Root Markdown files describe setup, results, licensing, third-party data, security, portability, and data-availability coverage. Inventories and hashes make every included file traceable.

