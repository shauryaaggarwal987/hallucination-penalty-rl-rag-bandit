# Final Reproducibility Report

- Frozen question, chunk, and FAISS hashes match the authoritative values.
- Permanent split integrity: 34 train, 7 validation, 9 test; no ID overlap or normalized-question duplicates.
- Corpus: 98 documents and 1,245 overlapping chunks. The overlap-inclusive chunk word count is 250,935.
- Six corrected runs each contain exactly 200 unique episodes, 200 updates, one run ID, final epsilon 0.02679593497159238, a completion marker, and no failure marker.
- Final training uses one generation and one selected-arm update per episode; API failures are not scored or updated.
- Regression suite: 18 tests passed.
- NLI response IDs use the scalar `row["repeat"]`; malformed bound-method IDs found: 0.
- Manifest excludes its own hash to avoid a stale self-reference.
- Historical reward wording: 254 is the primary-plus-baseline subset; 1,254 is all six historical training runs plus the baseline.
