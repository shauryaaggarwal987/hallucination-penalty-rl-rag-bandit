# Third-Party Data

## Wikipedia-derived experiment corpus

- Source: English Wikipedia API, as implemented in the authoritative notebook.
- Included publicly: document identifiers and corpus/chunk statistics in `data/corpus_manifest.csv`; reconstruction code in the notebook.
- Raw text redistributed: no.
- Processed text/index redistributed: no; the chunks reproduce the underlying article text.
- Reason: the frozen manifest lacks complete per-article source URL, revision, and attribution fields. This release does not invent missing provenance or license records.
- Reconstruction: execute the notebook's corpus acquisition cells against the Wikipedia API, preserve the returned page/revision metadata, then rebuild the index.

Wikipedia content is generally subject to Wikimedia terms and page-specific attribution/history requirements. Researchers rebuilding the corpus are responsible for recording and satisfying the terms that apply to the retrieved revisions.

## Project-authored sample documents

Three small instructional documents about RAG, reinforcement learning, and FAISS are included. They originate from the working project rather than the external corpus. Their reuse remains subject to the repository's currently unspecified software/content license.



## Public-release context sanitization

The 95-document third-party corpus and text-bearing processed artifacts are not redistributed. Public experiment artifacts retain generated responses, source IDs, retrieval order/similarities where recorded, context hashes, scores, and metrics; full retrieved-context text has been removed. These artifacts support result auditability, while full retrieval/LLM re-execution requires reconstructing the corpus and FAISS index.

