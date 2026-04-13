# Phase 5 Experiments Index

This repository copy keeps the **latest final Phase 5 view** in the main `results/` directory. The goal of this index is to point collaborators directly to the canonical final packages and their supporting statistics.

## Recommended final packages

- `results/phase5_with_recommendation_Hybrid_final/`: final Hybrid view combining unified main metrics + p500 perturbation + stats
- `results/phase5_with_recommendation_Retrieval_Only_final/`: final Retrieval-only view combining unified main metrics + p500 perturbation + stats
- `results/phase5_with_recommendation_KG_Only_final/`: final KG-only view combining unified main metrics + p500 perturbation + stats
- `results/phase5_stats/`: paired significance tests and bootstrap summaries for the finalized Phase 5 comparisons

## Included Phase 5 code

- `rag/`: retrieval, generation, pipeline, and faithfulness evaluation code
- `analysis/phase5_significance.py`: significance testing for main comparisons and perturbation follow-ups
- `scripts/`: reusable Phase 5 run scripts, including unified-BERTScore reevaluation and p500 perturbation follow-ups

## Included Phase 5 inputs

- `results/recommendations_v2.csv`
- `results/recommendations_v4.csv`
- `data/kg/kg_paths_for_recommendations.json`

