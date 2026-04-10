# phase5_with_recommendation_KG_Only

Official full-scale KG-only ablation package. This experiment isolates structured KG-path evidence without retrieval passages, using the same `recommendations_v4` input as the other Phase 5 experiments.

This directory stores a compact Phase 5 result package suitable for the GitHub working copy. Large raw explanation JSONL files remain in the AutoDL experiment workspace and are referenced through the source manifest.

## Full-run summary

- KG-only count: 59481
- Prompt-only companion count: 59481
- KG-only metrics: Overlap 0.2007, ROUGE-L 0.2112, Sem.Sim 0.9452, BERTScore 0.8835
- Prompt-only metrics: Overlap 0.0441, ROUGE-L 0.1063, Sem.Sim 0.9695, BERTScore 0.8279
- Perturbation E4: Overlap 0.0650, ROUGE-L 0.1108, Sem.Sim 0.9380, BERTScore 0.8424
