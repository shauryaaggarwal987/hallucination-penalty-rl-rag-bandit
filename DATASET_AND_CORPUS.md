# Dataset and Corpus

The frozen question bank contains 50 unique IDs split permanently into 34 training, 7 validation, and 9 test questions. The manifests contain no cross-split ID or normalized-question overlap and do not add answerability labels.

The experiment corpus contains 98 documents: three project sample documents and 95 Wikipedia-derived documents. Chunking produced 1,245 overlapping chunks and an overlap-inclusive chunk word count of 250,935. This is not a unique-corpus word count.

Raw Wikipedia-derived text and text-bearing index artifacts are excluded from this public copy for the provenance reason in `THIRD_PARTY_DATA.md`. `data/corpus_manifest.csv` lists frozen identifiers, inferred article titles, source category, chunk counts, and overlap-inclusive chunk word counts. Empty URL/license fields mean the information was not preserved; they must not be read as an attribution claim.

To reconstruct, use the Wikipedia API acquisition cells in `notebooks/REVISED_ALL_IN_ONE_hallucination_penalty.ipynb`, record canonical page URLs and revision identifiers, write the documents beneath `data/wikipedia_corpus/`, then run `python rag/build_index.py`. A matching build should contain 98 documents and 1,245 chunks and match the hashes in `artifacts/hashes/INPUT_INTEGRITY_REPORT.json`.



## Public-release context sanitization

The 95-document third-party corpus and text-bearing processed artifacts are not redistributed. Public experiment artifacts retain generated responses, source IDs, retrieval order/similarities where recorded, context hashes, scores, and metrics; full retrieved-context text has been removed. These artifacts support result auditability, while full retrieval/LLM re-execution requires reconstructing the corpus and FAISS index.

