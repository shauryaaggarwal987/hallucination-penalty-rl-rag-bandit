# Historical Post-hoc Rescore Summary

This is a post-hoc reward rescore of fixed historical trajectories, not corrected online training. All 254 scores in the historical primary-plus-baseline subset reproduced their stored rewards exactly; the wider six-training-run-plus-baseline audit covered 1,254 scores.

- Primary: 2 of 200 answers changed; mean 0.548571 -> 0.548333.
- Baseline: 4 of 54 answers changed; 6 formatting units removed.

| Policy | Legacy mean | Clean-v2 mean | Delta |
|---|---:|---:|---:|
| Citation-Enforced-RAG | 0.555556 | 0.555556 | +0.000000 |
| Fixed-A0 | 0.444444 | 0.444444 | +0.000000 |
| Fixed-A1 | 0.444444 | 0.444444 | +0.000000 |
| Fixed-A2 | 0.555556 | 0.555556 | +0.000000 |
| Fixed-A3 | 0.277778 | 0.555556 | +0.277778 |
| Fixed-A4 | 0.205556 | 0.251852 | +0.046296 |

A3 was disproportionately affected (0.277778 -> 0.555556). A4 increased from 0.205556 to 0.251852. Citation-Enforced, A0, A1, and A2 means were unchanged.
