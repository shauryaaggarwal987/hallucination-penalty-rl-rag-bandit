# Corrected Results Handoff

All values below come from newly generated corrected experiments unless explicitly labeled historical.

- Corrected primary run ID: `corrected_seed42_20260722_195822_697960`
- Seed: 42
- Final greedy action: A2
- Mean training reward: 0.556143
- Final action counts: `{'A0': 3, 'A1': 15, 'A2': 173, 'A3': 5, 'A4': 4}`
- Final Q-values: `{'A0': 0.3333333333333333, 'A1': 0.4666666666666666, 'A2': 0.5722543352601155, 'A3': 0.4, 'A4': 0.5571428571428572}`

## Corrected held-out policy results

| Policy | Mean reward | Abstention rate | Answer-only unsupported rate | Mean NLI support |
|---|---:|---:|---:|---:|
| Citation-Enforced-RAG | 0.444444 | 0.333333 | 0.166667 | 0.750000 |
| Fixed-A0 | 0.333333 | 0.555556 | 0.125000 | 0.750000 |
| Fixed-A1 | 0.555556 | 0.444444 | 0.000000 | 0.800000 |
| Fixed-A2 | 0.666667 | 0.333333 | 0.000000 | 0.833333 |
| Fixed-A3 | 0.555556 | 0.444444 | 0.000000 | 0.950000 |
| Fixed-A4 | 0.353744 | 0.000000 | 0.323128 | 0.432418 |
| Generate-All-Score-Max | 0.726496 | 0.222222 | 0.032967 | 0.769231 |
| Primary-Bandit-Greedy | 0.666667 | 0.333333 | 0.000000 | 0.833333 |
| Uniform-Random-Expected | 0.492971 | 0.355556 | 0.063095 | nan |
| Validation-Selected-Best-Fixed | 0.666667 | 0.333333 | 0.000000 | 0.833333 |

## Validation selection

A2 and A3 tied on validation; `A2` was selected by the predefined lower-temperature tie-break.

## Automatic NLI agreement

Across 430 substantive scoring units, raw agreement was 0.7163 and Cohen’s kappa was 0.3559. These compare two automatic metrics, not truth.

## Historical provenance

The historical primary-plus-baseline subset contains 254 reproduced scores; all six historical training runs plus the baseline contain 1,254 reproduced scores. Historical clean-v2 rescoring is stored only under `historical_rescored/` and `legacy_rescore/` and is not a corrected online result.
