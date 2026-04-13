# phase5_with_recommendation_Hybrid_final

Canonical final package for **Hybrid KG+RAG**.

This final view combines:
- the **main comparison** from the unified-BERTScore Phase 5.4 reevaluation,
- the **perturbation analysis** from the larger `p500` follow-up, and
- the latest **significance testing** outputs.

## Main comparison

| Condition | Count | Overlap | ROUGE-L | Sem.Sim | BERTScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid KG+RAG | 59500 | 0.2024 | 0.2345 | 0.9088 | 0.8443 |
| Prompt-only companion | 59500 | 0.0968 | 0.1705 | 0.9462 | 0.8135 |

## Perturbation (`p500`)

| Condition | Count | Overlap | ROUGE-L | Sem.Sim | BERTScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 | 500 | 0.1962 | 0.2311 | 0.9073 | 0.8434 |
| E2 | 500 | 0.1226 | 0.1631 | 0.9061 | 0.8367 |
| E3 | 500 | 0.1871 | 0.2222 | 0.9072 | 0.8420 |
| E4 | 500 | 0.0361 | 0.1432 | 0.9115 | 0.7959 |

## Provenance

- Main comparison source: `/root/autodl-tmp/github_codes/results/phase5_with_recommendation_Hybrid_bertscore_unified`
- Perturbation source: `/root/autodl-tmp/github_codes/results/phase5_with_recommendation_Hybrid_p500`
- Significance source: `/root/autodl-tmp/github_codes/results/phase5_stats`

Read `significance_summary.md` for the full statistical report and `significance_excerpt.json` for the subset most relevant to this experiment line.
