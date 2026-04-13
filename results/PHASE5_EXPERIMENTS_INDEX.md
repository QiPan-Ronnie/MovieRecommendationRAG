# Phase 5 Experiments Index

This repository copy keeps the **latest final Phase 5 view** in the main `results/` directory. Older packaged views are preserved locally under `results/_archive_legacy/` and are ignored by Git.

## Recommended final packages

- `results/phase5_with_recommendation_Hybrid_final/`: final Hybrid view combining unified main metrics + p500 perturbation + stats
- `results/phase5_with_recommendation_Retrieval_Only_final/`: final Retrieval-only view combining unified main metrics + p500 perturbation + stats
- `results/phase5_with_recommendation_KG_Only_final/`: final KG-only view combining unified main metrics + p500 perturbation + stats
- `results/phase5_stats/`: paired significance tests and bootstrap summaries for the finalized Phase 5 comparisons

## Included Phase 5 code

- `rag/`: retrieval, generation, pipeline, and faithfulness evaluation code
- `analysis/phase5_significance.py`: significance testing for main comparisons and perturbation follow-ups
- `scripts/`: reusable Phase 5 run scripts, including unified-BERTScore reevaluation and p500 perturbation follow-ups
- `tests/test_phase5_modes.py`
- `tests/test_faithfulness_config.py`
- `tests/test_phase5_significance.py`

## Included Phase 5 inputs

- `results/results_from_kg/recommendations_v2.csv`
- `results/results_from_kg/recommendations_v4.csv`
- `data/kg/kg_paths_for_recommendations.json`

## Legacy packages

Older packaging layers remain locally in `results/_archive_legacy/` for traceability, but they are no longer part of the public-facing final result layout.
