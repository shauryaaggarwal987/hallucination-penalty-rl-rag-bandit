# GitHub Release Validation Report

| Check | Status |
|---|---|
| Python syntax/import-independent parse | PASS |
| Regression suite | PASS — 18 tests |
| Authoritative public notebook | PASS — sanitized copy present |
| Question bank and split | PASS — 50 IDs; 34/7/9 split; no overlap |
| Historical primary and baseline | PASS |
| Corrected runs | PASS — six `RUN_COMPLETE.json`, zero `RUN_FAILED.json` |
| Raw third-party corpus files | PASS — not included |
| Raw retrieved-context text | PASS — sanitized from public result artifacts and notebook outputs |
| Corrected-result provenance | PASS — historical-rescored statistic moved to `results/historical/rescored/statistics/` |
| Statistics consistency | PASS — corrected bootstrap regenerated; paired bootstrap verified |
| Secret and private-material scan | PASS |
| SHA/inventory/path-map freshness | PASS — regenerated after sanitization |

Overall: **READY FOR PUBLIC GITHUB UPLOAD**, subject to the owner selecting a software license and independently reviewing the release before publication.

