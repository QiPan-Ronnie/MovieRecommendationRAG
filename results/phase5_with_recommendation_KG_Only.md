# phase5_with_recommendation_KG_Only

Official full-scale KG-only ablation built on the same `recommendations_v4` input as the other Phase 5 experiments.

## Actual artifact locations

- Results: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only`
- Logs: `/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendation_KG_Only`
- Scripts: `/root/autodl-tmp/MovieRecommendationRAG/scripts/phase5_with_recommendations_KG_Only`

## Phase mapping

- Phase 5.2 generation:
  - KG-only file: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only/explanations_kg_only.jsonl`
  - Prompt-only companion: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only/explanations_prompt_only.jsonl`
  - Log file: `/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendation_KG_Only/phase52.log`
- Phase 5.3 perturbation:
  - Result file: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only/perturbation_results.jsonl`
  - Log file: `/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendation_KG_Only/phase53.log`
  - Master orchestration log: `/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendation_KG_Only/master_phase53_phase54_hf.log`
- Phase 5.4 faithfulness:
  - KG-only summary: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only/faithfulness_kg_only/faithfulness_summary.json`
  - Prompt-only summary: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only/faithfulness_prompt_only/faithfulness_summary.json`
  - Perturbation summary: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only/faithfulness_perturbation/faithfulness_summary.json`
  - Log file: `/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendation_KG_Only/phase54.log`

## Final numbers

| Condition | Count | Overlap | ROUGE-L | Sem.Sim | BERTScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| KG-only | 59481 | 0.2007 | 0.2112 | 0.9452 | 0.8835 |
| Prompt-only | 59481 | 0.0441 | 0.1063 | 0.9695 | 0.8279 |

## Perturbation summary

| Condition | Count | Overlap | ROUGE-L | Sem.Sim | BERTScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 | 200 | 0.2012 | 0.2060 | 0.9450 | 0.8809 |
| E2 | 200 | 0.2040 | 0.2234 | 0.9452 | 0.8818 |
| E3 | 200 | 0.2005 | 0.2021 | 0.9471 | 0.8812 |
| E4 | 200 | 0.0650 | 0.1108 | 0.9380 | 0.8424 |

## Interpretation

This experiment provides the missing ablation needed to compare `prompt-only`, `retrieval-only`, `KG-only`, and `hybrid KG+RAG` under the same recommendation setup.
