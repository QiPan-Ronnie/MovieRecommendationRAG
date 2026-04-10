# phase5_with_recommendation_Retrieval_Only

Historical retrieval-only baseline derived from the run where KG path injection did not take effect.

## Actual artifact locations

- Results root: `/root/autodl-tmp/MovieRecommendationRAG/results/backup_pre_kg_fix_20260403_152531`
- Logs root: `/root/autodl-tmp/MovieRecommendationRAG/logs/backup_pre_kg_fix_20260403_152531`

## Phase mapping

- Phase 5.2 generation:
  - RAG file: `/root/autodl-tmp/MovieRecommendationRAG/results/backup_pre_kg_fix_20260403_152531/phase5_with_recommendations_v4/explanations_rag.jsonl`
  - Prompt-only file: `/root/autodl-tmp/MovieRecommendationRAG/results/backup_pre_kg_fix_20260403_152531/phase5_with_recommendations_v4/explanations_prompt_only.jsonl`
  - Main log: `/root/autodl-tmp/MovieRecommendationRAG/logs/backup_pre_kg_fix_20260403_152531/phase5_with_recommendations_v4/phase52.log`
- Phase 5.3 perturbation:
  - Result file: `/root/autodl-tmp/MovieRecommendationRAG/results/backup_pre_kg_fix_20260403_152531/RAG_Phase_5.3/perturbation_results.jsonl`
  - Log files:
    - `/root/autodl-tmp/MovieRecommendationRAG/logs/backup_pre_kg_fix_20260403_152531/phase53_v4.log`
    - `/root/autodl-tmp/MovieRecommendationRAG/logs/backup_pre_kg_fix_20260403_152531/phase5_with_recommendations_v4/phase53_nohup.log`
- Phase 5.4 faithfulness:
  - RAG summary: `/root/autodl-tmp/MovieRecommendationRAG/results/backup_pre_kg_fix_20260403_152531/phase5_with_recommendations_v4/faithfulness_rag/faithfulness_summary.json`
  - Prompt-only summary: `/root/autodl-tmp/MovieRecommendationRAG/results/backup_pre_kg_fix_20260403_152531/phase5_with_recommendations_v4/faithfulness_prompt_only/faithfulness_summary.json`
  - Perturbation summary: `/root/autodl-tmp/MovieRecommendationRAG/results/backup_pre_kg_fix_20260403_152531/phase5_with_recommendations_v4/faithfulness_perturbation/faithfulness_summary.json`
  - Log file: `/root/autodl-tmp/MovieRecommendationRAG/logs/backup_pre_kg_fix_20260403_152531/phase5_with_recommendations_v4/phase54.log`

## Interpretation

Use this as the current retrieval-only RAG baseline, but note that it originated from the pre-fix run rather than an explicit ablation script.
