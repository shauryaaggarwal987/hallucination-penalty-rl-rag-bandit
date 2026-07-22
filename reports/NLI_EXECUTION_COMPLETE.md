# NLI and SBERT Execution Completed

The temporary dependency blocker was resolved. Exact SBERT rescoring, retrieval diagnostics, threshold sensitivity, and context-conditioned BART-MNLI evaluation were executed and saved. This file is retained only to record that the earlier blocker no longer applies.

The NLI evaluator used each actual retrieved chunk as premise and each cleaned substantive scoring unit as hypothesis with `facebook/bart-large-mnli`, fixed threshold 0.50, and maximum entailment across the five chunks. Results remain automatic-proxy evidence, not factual truth.
