# phase5_with_recommendation_KG_Only_final

Canonical final package for **KG-only**.

This final view combines:
- the **main comparison** from the unified-BERTScore Phase 5.4 reevaluation,
- the **perturbation analysis** from the larger `p500` follow-up, and
- the latest **significance testing** outputs.

## Main comparison

| Condition | Count | Overlap | ROUGE-L | Sem.Sim | BERTScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| KG-only | 59481 | 0.2007 | 0.2112 | 0.9452 | 0.8835 |
| Prompt-only companion | 59481 | 0.0441 | 0.1063 | 0.9695 | 0.8279 |

## Perturbation (`p500`)

| Condition | Count | Overlap | ROUGE-L | Sem.Sim | BERTScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 | 500 | 0.1984 | 0.2125 | 0.9458 | 0.8833 |
| E2 | 500 | 0.1928 | 0.1929 | 0.9551 | 0.8869 |
| E3 | 500 | 0.1921 | 0.1977 | 0.9461 | 0.8797 |
| E4 | 500 | 0.0642 | 0.1032 | 0.9372 | 0.8357 |

## Notes

This package should be read as the canonical **KG-only** summary for Phase 2.
- `main_summary.json` reports the unified-BERTScore main comparison.
- `perturbation_summary.json` reports the larger `p500` perturbation follow-up.
- `significance_summary.md` and `significance_excerpt.json` summarize the paired statistical comparisons most relevant to this experiment line.
