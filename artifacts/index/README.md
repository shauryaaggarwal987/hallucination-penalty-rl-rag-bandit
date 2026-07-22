# Frozen Index Status

The experiments used `sentence-transformers/all-MiniLM-L6-v2`, a FAISS `IndexFlatIP`, 98 documents, 1,245 overlapping chunks, and retrieval top-k 5.

The binary index and text-bearing `chunks.jsonl`/`chunk_metadata.json` are not redistributed in this public package because they encode the complete third-party article text and the available provenance does not preserve sufficient per-revision attribution for confident redistribution. Their authoritative hashes are recorded in `../hashes/INPUT_INTEGRITY_REPORT.json`.

Rebuild the corpus using the authoritative notebook's Wikipedia API cells, then run `python rag/build_index.py`. See `DATASET_AND_CORPUS.md` for the expected counts and limitations.

