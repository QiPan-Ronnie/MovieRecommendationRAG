# Phase 5 Experiments Index

This repository copy contains the Phase 5 code, scripts, tests, key inputs, and compact result packages. Large raw explanation artifacts remain in the AutoDL experiment workspace and are referenced via source manifests inside each experiment directory.

## Canonical experiment directories

- `results/phase5_with_recommendation_Hybrid/`: corrected full Hybrid KG+RAG package
- `results/phase5_with_recommendation_Retrieval_Only/`: retrieval-only baseline package
- `results/phase5_with_recommendation_Prompt_Only/`: prompt-only baseline package with companion summaries from multiple experiment lines
- `results/phase5_with_recommendation_KG_Only/`: completed full KG-only ablation package

## Included Phase 5 code

- `rag/`: retrieval, generation, pipeline, and faithfulness evaluation code
- `scripts/`: reusable Phase 5 run scripts, including Hybrid and KG-only variants
- `tests/test_phase5_modes.py`: KG-only regression tests

## Included Phase 5 inputs

- `results/results_from_kg/recommendations_v2.csv`
- `results/results_from_kg/recommendations_v4.csv`
- `data/kg/kg_paths_for_recommendations.json`
