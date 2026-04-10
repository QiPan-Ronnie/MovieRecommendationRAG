# phase5_with_recommendation_Hybrid

Official corrected hybrid experiment using `recommendations_v4 + KG path + retrieval evidence`.

## Actual artifact locations

- Results: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendations_v4&KG_Path`
- Logs: `/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendations_v4&KG_Path`

## Phase mapping

- Phase 5.2 generation:
  - Result file: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendations_v4&KG_Path/explanations_rag.jsonl`
  - Prompt-only companion: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendations_v4&KG_Path/explanations_prompt_only.jsonl`
  - Log file: `/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendations_v4&KG_Path/phase52.log`
- Phase 5.3 perturbation:
  - Result file: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendations_v4&KG_Path/perturbation_results.jsonl`
  - Log file: `/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendations_v4&KG_Path/phase53.log`
- Phase 5.4 faithfulness:
  - RAG summary: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendations_v4&KG_Path/faithfulness_rag/faithfulness_summary.json`
  - Prompt-only summary: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendations_v4&KG_Path/faithfulness_prompt_only/faithfulness_summary.json`
  - Perturbation summary: `/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendations_v4&KG_Path/faithfulness_perturbation/faithfulness_summary.json`
  - Log file: `/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendations_v4&KG_Path/phase54.log`

## Interpretation

This is the main full-scale hybrid run and should be the default source for final `KG+RAG` numbers.
