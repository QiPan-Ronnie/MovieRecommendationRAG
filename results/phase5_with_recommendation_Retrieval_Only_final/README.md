# phase5_with_recommendation_Retrieval_Only_final

Canonical final package for **Retrieval-only RAG**.

This final view combines:
- the **main comparison** from the unified-BERTScore Phase 5.4 reevaluation,
- the **perturbation analysis** from the larger `p500` follow-up, and
- the latest **significance testing** outputs.

## Main comparison

| Condition | Count | Overlap | ROUGE-L | Sem.Sim | BERTScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retrieval-only RAG | 59500 | 0.1452 | 0.1674 | 0.8871 | 0.8391 |
| Prompt-only companion | 59500 | 0.0861 | 0.1368 | 0.9244 | 0.8196 |

## Perturbation (`p500`)

| Condition | Count | Overlap | ROUGE-L | Sem.Sim | BERTScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 | 500 | 0.1354 | 0.1780 | 0.8976 | 0.8359 |
| E2 | 500 | 0.0979 | 0.1345 | 0.9107 | 0.8389 |
| E3 | 500 | 0.1369 | 0.1776 | 0.8980 | 0.8355 |
| E4 | 500 | 0.0274 | 0.1191 | 0.9115 | 0.8008 |

## Notes

This package should be read as the canonical **Retrieval-only RAG** baseline for Phase 2.
- `main_summary.json` reports the unified-BERTScore main comparison.
- `perturbation_summary.json` reports the larger `p500` perturbation follow-up.
- `significance_summary.md` and `significance_excerpt.json` summarize the paired statistical comparisons most relevant to this experiment line.
