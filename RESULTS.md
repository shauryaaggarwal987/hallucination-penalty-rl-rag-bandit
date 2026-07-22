# Verified Results

The corrected primary run is `corrected_seed42_20260722_195822_697960`: 200 episodes, mean reward 0.556143, final greedy action A2, and 200 updates. Five additional corrected seed runs are included with their completion markers.

Held-out mean rewards include Fixed-A2/primary greedy 0.666667, citation-enforced RAG 0.444444, Fixed-A4 0.353744, uniform-random expected 0.492971, and generate-all score-max 0.726496. Generate-all is an oracle-like five-call evaluation comparator, not bandit training. Validation selected A2 after an A2/A3 tie using the predefined lower-temperature rule.

Across 430 substantive scoring units, raw semantic/NLI agreement was 0.7163 and Cohen's kappa was 0.3559. Both measures are automatic proxies rather than factual ground truth.

Exact tables, abstention-aware metrics, bootstrap intervals, retrieval diagnostics, and computational comparison are in `results/corrected/`. Historical outputs and clean-v2 post-hoc rescoring are in `results/historical/` and must not be interpreted as fresh corrected trajectories.

