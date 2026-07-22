# Final Non-LaTeX Completion Report

## Status

All authorized technical workflows completed: six corrected 200-episode training runs, the native held-out matrix, three-repeat validation selection, three-repeat controlled-temperature ablation, new-response NLI, abstention-aware summaries, fixed-threshold sensitivity, retrieval diagnostics, paired bootstrap statistics, computational comparison, figures, tests, and manifest generation.

## Primary result

The corrected seed-42 run is `corrected_seed42_20260722_195822_697960`. Its final greedy action is A2 and its mean training reward is 0.556143.

On the nine-question test matrix, the primary greedy policy (A2) has mean clean-v2 reward 0.666667, abstention rate 0.333333, answer-only unsupported-unit rate 0, and mean answer-only NLI support 0.833333. The validation-selected strategy is also A2 and therefore has the same cached held-out responses. Generate-All-Score-Max reaches 0.726496 but requires five generation calls per query and is oracle-like.

## Multi-seed result

Across six corrected runs, the mean of per-seed training means is 0.542815 (SD 0.009558). Final greedy actions are A2 for three seeds, A0 for two, and A3 for one. This is descriptive seed sensitivity, not proof of convergence.

## Metric language and corpus count

The NLI analysis covers 430 substantive scoring units. It is an automatic context-entailment proxy, not factual truth. The value 250,935 is the overlap-inclusive sum of chunk word counts, not the unique-corpus word count.

## Scope

Historical frozen-response rescores are clearly segregated and are not called corrected online results. Human annotation and manuscript/LaTeX work were not performed.
