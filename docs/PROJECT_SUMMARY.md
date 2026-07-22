# Corrected Technical Project Summary

This project studies global generation-strategy selection for RAG using a non-contextual epsilon-greedy bandit. It does not implement contextual per-query routing and does not fine-tune model weights.

## Current design

- Gemini model: `gemini-2.5-flash`.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Semantic threshold: 0.50.
- Retrieval: frozen FAISS index, top-k 5, 1,245 chunks from 98 documents.
- Training: 200 episodes; initial epsilon 0.20, decay 0.99, floor 0.01.
- Arms: A0-A4 with native temperatures 0.0, 0.2, 0.0, 0.2, and 0.9.
- Correct protocol: choose one arm, generate one answer, score one answer, update one arm.
- Corrected scorer: `semantic_proxy_clean_v2`; formatting-only list fragments are excluded and citation tags are removed only from semantic embeddings.
- NLI: chunk-wise premise-hypothesis BART-MNLI evaluation, kept independent of training.

The semantic reward and NLI output are automatic support proxies, not factual truth. Abstention, answer rate, citation metrics, and unsupported-unit rates must be interpreted together.

Historical outputs document the earlier implementation and are frozen. Corrected online results require fresh online reruns because post-hoc rescoring cannot reproduce the changed action trajectory.
