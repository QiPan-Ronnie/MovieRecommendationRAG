"""
End-to-end pipeline: Phase 0-3.
Run this on the server with GPU.

Usage:
    python run_all.py                    # Run everything
    python run_all.py --skip-tmdb        # Skip TMDB fetch (if already done)
    python run_all.py --phase 1          # Run only Phase 1 (baselines)
    python run_all.py --phase 2          # Run only Phase 2 (KG)
    python run_all.py --phase 3          # Run only Phase 3 (ranker + ablation)
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

    # Parse ML-1M
    print("\n[0.1] Parsing MovieLens 1M...")
    from src.data_prep.parse_ml1m import main as parse_main
    parse_main()

    # Fetch TMDB (if not skipping)
    if not skip_tmdb:
        print("\n[0.2] Fetching TMDB metadata...")
        print("  (This will use cached results via checkpoint/resume)")
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
    """Phase 2: KG construction and feature engineering."""
    print("\n" + "#" * 60)
    print("# Phase 2: Knowledge Graph")
    print("#" * 60)

    print("\n[2.1] Building KG from TMDB metadata...")
    from src.kg.build_kg import main as kg_main
    kg_main()

    print("\n[2.2] Computing content similarity (Sentence-Transformer)...")
    from src.kg.content_similarity import main as content_sim_main
    content_sim_main()

    print("\n[2.3] Computing KG features...")
    from src.kg.kg_features import main as kg_feat_main
    kg_feat_main()


def run_phase3(cf_scores_path="results/cf_scores.csv"):
    """Phase 3: Ranker and ablation."""
    print("\n" + "#" * 60)
    print("# Phase 3: Ranker + Ablation")
    print("#" * 60)

    # Auto-detect best baseline scores
    score_files = {
        "results/lightgcn_scores.csv": "lgcn_score",
        "results/mf_scores.csv": "mf_score",
        "results/cf_scores.csv": "cf_score",
    }

    # Use best available (prefer LightGCN > MF > CF)
    for path, _ in score_files.items():
        if os.path.exists(path):
            cf_scores_path = path
            break

    print(f"Using CF scores from: {cf_scores_path}")

    from src.ranker.ranker import run_ablation
    run_ablation(cf_scores_path=cf_scores_path)


def main():
    parser = argparse.ArgumentParser(description="Run experiment pipeline")
    parser.add_argument("--phase", type=int, help="Run specific phase (0, 1, 2, or 3)")
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
        else:
            print(f"Unknown phase: {args.phase}")
    else:
        # Run all
        run_phase0(skip_tmdb=args.skip_tmdb)
        run_phase1()
        run_phase2()
        run_phase3()

    print("\n" + "#" * 60)
    print("# DONE!")
    print("#" * 60)


if __name__ == "__main__":
    main()
