# Scorer Changelog

## `semantic_proxy_legacy_v1`

Frozen historical behavior used ordinary sentence splitting, could emit standalone numbering fragments, and retained `[source: ...]` citation strings in semantic embeddings. Strict abstention in the authoritative historical notebook was an exact `Not in context` match after terminal punctuation removal.

## `semantic_proxy_clean_v2`

- Uses a scoring-specific factual-unit extractor without changing document chunking.
- Removes pure numbering, bullets, citation-only, punctuation-only, and empty units.
- Retains substantive numbered and bulleted claims.
- Removes Markdown emphasis and citation tags from the copy embedded for semantic similarity.
- Preserves raw units and citations for citation validity and coverage.
- Computes citation coverage over substantive units only.
- Keeps threshold 0.50, reward formula, and strict abstention semantics unchanged.

Corrected online training must be rerun: changing observed rewards can change later epsilon-greedy decisions, so a rescored historical trajectory is only a counterfactual diagnostic.
