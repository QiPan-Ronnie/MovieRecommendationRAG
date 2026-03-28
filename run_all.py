"""
End-to-end pipeline: Phase 0-4.
Run this on the server with GPU.

Pipeline order:
  Phase 0:   Data preparation (parse, filter, split)
  Phase 1:   Baseline recall models (Item-CF, BPR-MF, LightGCN)
  Phase 2:   KG construction + TransE training
  Phase 2.5: Multi-route recall (CF + KG candidates)
  Phase 2.6: Build recall candidates (label with val/test)
  Phase 2.7: Feature engineering for recall candidates
  Phase 3:   Ranker ablation (V1/V2/V3/V3e/V4)
  Phase 4:   Long-tail analysis (RQ2)

Usage:
    python run_all.py                    # Run everything
    python run_all.py --skip-tmdb        # Skip TMDB fetch (if already done)
    python run_all.py --phase 1          # Run only Phase 1 (baselines)
    python run_all.py --phase 2          # Run only Phase 2 (KG + multi-recall + features)
    python run_all.py --phase 3          # Run only Phase 3 (ranker + ablation)
    python run_all.py --phase 4          # Run only Phase 4 (long-tail analysis)
"""
import argparse
import os
import sys
import subprocess


def run_phase0(skip_tmdb=False):
    """Phase 0: Data preparation."""
    print("\n" + "#" * 60)
    print("# Phase 0: Data Preparation")
    print("#" * 60)

    print("\n[0.1] Parsing MovieLens 1M...")
    from src.data_prep.parse_ml1m import main as parse_main
    parse_main()

    if not skip_tmdb:
        print("\n[0.2] Fetching TMDB metadata...")
        api_key = os.environ.get("TMDB_API_KEY")
        if not api_key:
            print("ERROR: TMDB_API_KEY environment variable not set.")
            print("  Set it with: export TMDB_API_KEY=your_key_here")
            sys.exit(1)
        subprocess.run([
            sys.executable, "src/data_prep/fetch_tmdb.py",
            "--api_key", api_key,
            "--delay", "0.05"
        ], check=True)
    else:
        print("\n[0.2] Skipping TMDB fetch (--skip-tmdb)")


def run_phase1():
    """Phase 1: Baseline models."""
    print("\n" + "#" * 60)
    print("# Phase 1: Baseline Models")
    print("#" * 60)

    from src.run_baselines import main as baselines_main
    baselines_main()


def run_phase2():
    """Phase 2: KG construction, TransE, multi-recall, candidate labeling, features."""
    print("\n" + "#" * 60)
    print("# Phase 2: Knowledge Graph + Multi-Route Recall + Features")
    print("#" * 60)

    # 2.1 Build KG
    print("\n[2.1] Building KG from TMDB metadata...")
    from src.kg.build_kg import main as kg_main
    kg_main()

    # 2.2 Train TransE
    print("\n[2.2] Training TransE KG embeddings...")
    from src.kg.transe import main as transe_main
    transe_main()

    # 2.3 Multi-route recall (best recall model + KG)
    print("\n[2.3] Generating multi-route recall candidates...")
    from src.models.multi_recall import generate_multi_recall
    generate_multi_recall()  # auto-detects best recall model (LightGCN > MF > CF)

    # 2.4 Build recall candidate labels (val for training, test for evaluation)
    print("\n[2.4] Labeling recall candidates for ranker train/val/test...")
    scores_path = "results/multi_recall_scores.csv"
    from src.ranker.ranker import (
        build_recall_test_candidates,
        build_recall_train_val_candidates,
    )
    build_recall_test_candidates(
        cf_scores_path=scores_path,
        test_path="data/processed/test.csv",
        output_path="data/processed/test_recall_candidates.csv",
    )
    build_recall_train_val_candidates(
        cf_scores_path=scores_path,
        val_path="data/processed/val.csv",
        train_output="data/processed/train_recall_candidates.csv",
        val_output="data/processed/val_recall_candidates.csv",
    )

    # 2.5 Feature engineering for recall candidates
    print("\n[2.5] Computing content similarity (Sentence-Transformer)...")
    from src.kg.content_similarity import main as content_sim_main
    content_sim_main()

    print("\n[2.6] Computing KG features (hand-crafted)...")
    from src.kg.kg_features import main as kg_feat_main
    kg_feat_main()

    print("\n[2.7] Computing KG embedding features (TransE)...")
    from src.kg.kg_embedding_features import main as kg_emb_main
    kg_emb_main()


def run_phase3():
    """Phase 3: Ranker and ablation with distribution-matched training."""
    print("\n" + "#" * 60)
    print("# Phase 3: Ranker + Ablation (Distribution-Matched)")
    print("#" * 60)

    scores_path = "results/multi_recall_scores.csv"
    if not os.path.exists(scores_path):
        scores_path = "results/cf_scores.csv"

    print(f"Using recall scores from: {scores_path}")

    from src.ranker.ranker import run_ablation_matched
    run_ablation_matched(cf_scores_path=scores_path, do_hp_search=True)


def run_phase4():
    """Phase 4: Long-tail analysis (RQ2)."""
    print("\n" + "#" * 60)
    print("# Phase 4: Long-tail Analysis")
    print("#" * 60)

    from src.evaluation.longtail_analysis import run_longtail_analysis
    run_longtail_analysis()


def main():
    parser = argparse.ArgumentParser(description="Run experiment pipeline")
    parser.add_argument("--phase", type=int, help="Run specific phase (0-4)")
    parser.add_argument("--skip-tmdb", action="store_true", help="Skip TMDB fetch")
    args = parser.parse_args()

    if args.phase is not None:
        if args.phase == 0:
            run_phase0(skip_tmdb=args.skip_tmdb)
        elif args.phase == 1:
            run_phase1()
        elif args.phase == 2:
            run_phase2()
        elif args.phase == 3:
            run_phase3()
        elif args.phase == 4:
            run_phase4()
        else:
            print(f"Unknown phase: {args.phase}")
    else:
        run_phase0(skip_tmdb=args.skip_tmdb)
        run_phase1()
        run_phase2()
        run_phase3()
        run_phase4()

    print("\n" + "#" * 60)
    print("# DONE!")
    print("#" * 60)


if __name__ == "__main__":
    main()
