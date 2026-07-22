# Raw-Corpus Text Public-Release Audit

Status: **PASS**

The public release was audited recursively for embedded retrieved third-party corpus passages. Source IDs, article titles, context hashes, similarity scores, citations, generated responses, and numerical metrics remain; long retrieved evidence text was removed.

## Sanitized files

- `results/corrected/primary/corrected_seed42_20260722_195822_697960/context_snapshots.jsonl`
- `results/corrected/multi_seed/*/context_snapshots.jsonl` (five runs)
- `results/corrected/baselines/test_generation_matrix.csv`
- `results/corrected/baselines/test_scored_responses.csv`
- `results/corrected/validation_best_fixed/validation_generation_matrix.csv`
- `results/corrected/validation_best_fixed/offline_best_fixed_test_results.csv`
- `results/corrected/controlled_temperature/controlled_temperature_raw.csv`
- `notebooks/REVISED_ALL_IN_ONE_hallucination_penalty.ipynb`

Removed fields: `retrieved_chunk_texts` from context snapshots and `retrieved_contexts` from response-matrix files. One long text-bearing notebook output was removed. No generated answer, source-ID list, context hash, rank/similarity metadata, score, reward, NLI numerical result, or completion marker was removed.

NLI result files were inspected. They retain scoring units and best source IDs but do not store raw premises/retrieved chunks. Historical-rescore files retain only hashes and source IDs, not retrieved chunk text.

Known remaining text-bearing material consists of project-authored sample documents, question text, generated responses, source identifiers, and documentation—not raw Wikipedia-derived corpus passages.

