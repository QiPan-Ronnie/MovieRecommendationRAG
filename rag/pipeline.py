"""
RAG Pipeline Orchestrator — Stage 3 of the recommendation system.

End-to-end pipeline:
  Phase 5.1: Build RAG evidence corpus & indices
  Phase 5.2: Generate explanations (RAG + prompt-only) for top-K recommendations
  Phase 5.3: Run faithfulness perturbation experiments (E1-E4)
  Phase 5.4: Evaluate and report faithfulness metrics

Usage:
    python -m rag.pipeline                          # Run full RAG pipeline
    python -m rag.pipeline --phase 5.1              # Build corpus only
    python -m rag.pipeline --phase 5.2              # Generate explanations
    python -m rag.pipeline --phase 5.3              # Perturbation experiments
    python -m rag.pipeline --phase 5.4              # Evaluate faithfulness
    python -m rag.pipeline --llm-backend api        # Use API backend
    python -m rag.pipeline --api-url http://...     # Custom API URL
    python -m rag.pipeline --model Qwen/Qwen2.5-7B-Instruct  # Custom model
    python -m rag.pipeline --num-users 100          # Limit users for testing
"""
import argparse
import json
import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def _normalize_id(value) -> int:
    """Normalize pandas/numpy scalar IDs like 2858.0 to native ints."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            raise ValueError("ID value is NaN")
        if not float(value).is_integer():
            raise ValueError(f"Expected integral ID, got {value!r}")
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("ID value is empty")
        parsed = float(value)
        if not parsed.is_integer():
            raise ValueError(f"Expected integral ID, got {value!r}")
        return int(parsed)
    return int(value)


def _load_kg_paths(paths_file: str = "data/kg/kg_paths_for_recommendations.json",
                   required: bool = False,
                   context: str = "") -> dict:
    """Load KG structural paths keyed by "userId_movieId".""" 
    if not os.path.exists(paths_file):
        if required:
            label = f" for {context}" if context else ""
            raise FileNotFoundError(
                f"Missing KG paths file{label}: {paths_file}"
            )
        print("[INFO] No KG paths file found; skipping KG evidence augmentation")
        return {}

    print(f"Loading KG paths from {paths_file} ...")
    with open(paths_file, "r", encoding="utf-8") as f:
        kg_paths = json.load(f)
    print(f"  Loaded KG paths for {len(kg_paths)} (user, movie) pairs")
    return kg_paths


PRIMARY_MODE_BY_EVIDENCE_MODE = {
    "hybrid": "rag",
    "kg_only": "kg_only",
    "retrieval_only": "retrieval_only",
}

REFERENCE_MODE_PRIORITY = ("rag", "kg_only", "retrieval_only")
KG_MAX_PATHS = 3


def _get_primary_mode_label(evidence_mode: str) -> str:
    """Map an evidence configuration to its explanation file suffix."""
    if evidence_mode not in PRIMARY_MODE_BY_EVIDENCE_MODE:
        raise ValueError(f"Unsupported evidence_mode: {evidence_mode}")
    return PRIMARY_MODE_BY_EVIDENCE_MODE[evidence_mode]


def _build_evidence_bundle(retrieved_evidence: list[dict] | None,
                           kg_evidence: list[dict] | None,
                           evidence_mode: str) -> list[dict]:
    """Assemble the evidence bundle for a chosen ablation setting."""
    retrieved = list(retrieved_evidence or [])
    kg_only = list(kg_evidence or [])

    if evidence_mode == "hybrid":
        return kg_only + retrieved
    if evidence_mode == "kg_only":
        return kg_only
    if evidence_mode == "retrieval_only":
        return retrieved
    raise ValueError(f"Unsupported evidence_mode: {evidence_mode}")


def _pair_key(uid: int, mid: int) -> str:
    return f"{_normalize_id(uid)}_{_normalize_id(mid)}"


def _filter_recommendations_for_kg_paths(recs: pd.DataFrame,
                                         kg_paths: dict,
                                         evidence_mode: str) -> tuple[pd.DataFrame, int]:
    """Restrict KG-only experiments to recommendation pairs with KG paths."""
    if evidence_mode != "kg_only":
        return recs, 0

    if not kg_paths:
        raise ValueError("KG-only mode requires non-empty KG paths.")

    keys = recs.apply(lambda row: _pair_key(row["user_id"], row["movie_id"]), axis=1)
    mask = keys.isin(set(kg_paths.keys()))
    filtered = recs.loc[mask].copy()
    skipped = int((~mask).sum())
    if filtered.empty:
        raise ValueError("No recommendation pairs have KG paths for kg_only mode.")
    return filtered, skipped


def _pick_reference_explanation_path(output_dir: str) -> str | None:
    """Pick the strongest evidence-backed explanation file available."""
    for mode in REFERENCE_MODE_PRIORITY:
        candidate = os.path.join(output_dir, f"explanations_{mode}.jsonl")
        if os.path.exists(candidate):
            return candidate
    return None


def _iter_explanation_files(output_dir: str) -> list[tuple[str, str]]:
    """List explanation files in a stable evaluation order."""
    ordered_modes = list(REFERENCE_MODE_PRIORITY) + ["prompt_only"]
    files = []
    for mode in ordered_modes:
        path = os.path.join(output_dir, f"explanations_{mode}.jsonl")
        if os.path.exists(path):
            files.append((mode, path))
    return files


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_ranked_recommendations(path: str, k: int) -> pd.DataFrame | None:
    """Load a persisted top-K recommendation file if one is available."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    required = {"user_id", "movie_id"}
    if not required.issubset(df.columns):
        return None

    score_col = next(
        (c for c in ["pred_score", "score", "final_score", "rank_score", "cf_score"]
         if c in df.columns),
        None,
    )

    if "rank" in df.columns:
        df = df.sort_values(["user_id", "rank"], ascending=[True, True]).copy()
        df = df[df["rank"] <= k]
        if score_col:
            df["score"] = df[score_col]
        else:
            df["score"] = (k - df["rank"] + 1).astype(float)
        return df[["user_id", "movie_id", "rank", "score"]].reset_index(drop=True)

    if score_col:
        df = df.sort_values(["user_id", score_col], ascending=[True, False]).copy()
        df["rank"] = df.groupby("user_id").cumcount() + 1
        df = df[df["rank"] <= k]
        df["score"] = df[score_col]
        return df[["user_id", "movie_id", "rank", "score"]].reset_index(drop=True)

    return None


def get_top_k_recommendations(k: int = 10,
                               results_dir: str = "results") -> pd.DataFrame:
    """
    Load top-K recommended movies per user.

    Preference order:
      1. Persisted final re-ranking outputs, if available.
      2. Recall-stage scores as a fallback.

    Returns DataFrame with columns: user_id, movie_id, rank, score
    """
    preferred_paths = [
        # V4 LambdaMART output from teammates (main experiment, RQ1/RQ3)
        "results_from_kg/recommendations_v4.csv",
        "final_recommendations.csv",
        "reranked_recommendations.csv",
        "topk_recommendations.csv",
        "v4_lambdamart_recommendations.csv",
    ]
    for rel_path in preferred_paths:
        abs_path = os.path.join(results_dir, rel_path)
        ranked = _load_ranked_recommendations(abs_path, k=k)
        if ranked is not None:
            print(f"Loading final recommendations from {abs_path}")
            return ranked

    # Fallback: recall-stage ranking. Keep this explicit because it is not the
    # final LambdaMART recommendation list used in the main recommender.
    multi_path = os.path.join(results_dir, "multi_recall_scores.csv")
    if os.path.exists(multi_path):
        print("[WARN] No persisted final re-ranked recommendations found; "
              "falling back to recall-stage cf_score ordering.")
        df = pd.read_csv(multi_path)
        df = df.sort_values(["user_id", "cf_score"], ascending=[True, False])
        df["rank"] = df.groupby("user_id").cumcount() + 1
        df = df[df["rank"] <= k]
        df["score"] = df["cf_score"]
        return df[["user_id", "movie_id", "rank", "score"]].reset_index(drop=True)

    cf_path = os.path.join(results_dir, "cf_scores.csv")
    if os.path.exists(cf_path):
        print("[WARN] No final recommendation file found; using cf_scores.csv fallback.")
        df = pd.read_csv(cf_path)
        df = df.sort_values(["user_id", "cf_score"], ascending=[True, False])
        df["rank"] = df.groupby("user_id").cumcount() + 1
        df = df[df["rank"] <= k]
        df["score"] = df["cf_score"]
        return df[["user_id", "movie_id", "rank", "score"]].reset_index(drop=True)

    raise FileNotFoundError(
        "No recommendation scores found. Run Phase 1-2 first.")


def get_user_history(train_path: str = "data/processed/train.csv",
                     max_history: int = 10) -> dict[int, list[int]]:
    """Load user history (movie_ids) from training data."""
    train_df = pd.read_csv(train_path)
    if "timestamp" in train_df.columns:
        train_df = train_df.sort_values("timestamp")

    history = defaultdict(list)
    for _, row in train_df.iterrows():
        history[row["user_id"]].append(row["movie_id"])

    return {uid: mids[-max_history:] for uid, mids in history.items()}


def get_movie_info(movies_path: str = "data/processed/movies.csv",
                   tmdb_path: str = "data/tmdb/tmdb_metadata.csv") -> dict:
    """Load movie metadata: {movie_id: {"title", "genres", ...}}."""
    movies_df = pd.read_csv(movies_path)
    info = {}
    for _, row in movies_df.iterrows():
        info[row["movie_id"]] = {
            "title": row["title"],
            "genres": str(row.get("genres", "")),
        }

    if os.path.exists(tmdb_path):
        tmdb_df = pd.read_csv(tmdb_path)
        for _, row in tmdb_df.iterrows():
            mid = row["movie_id"]
            if mid in info:
                info[mid]["overview"] = str(row.get("overview", ""))
                info[mid]["actors"] = str(row.get("actors", ""))
                info[mid]["directors"] = str(row.get("directors", ""))

    return info


def _load_existing_results(path: str) -> set[tuple[int, int]]:
    """Load already-generated (user_id, movie_id) pairs from a JSONL file."""
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((_normalize_id(r["user_id"]), _normalize_id(r["movie_id"])))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def _load_existing_evidence_map(path: str) -> dict[tuple[int, int], list[dict]]:
    """Load evaluation reference evidence keyed by (user_id, movie_id)."""
    evidence_map = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    key = (_normalize_id(record["user_id"]), _normalize_id(record["movie_id"]))
                    evidence = record.get("reference_evidence")
                    if not evidence:
                        evidence = record.get("evidence_used", [])
                    evidence_map[key] = evidence
                except (json.JSONDecodeError, KeyError):
                    continue
    return evidence_map


def _make_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    return obj

# ---------------------------------------------------------------------------
# KG path evidence helpers
# ---------------------------------------------------------------------------

def _format_kg_path_text(path_obj: dict, movie_info: dict) -> str:
    """
    Convert a single KG path entry to a natural-language evidence string.

    path_obj example:
      {"history_movie": 2018,
       "path": [{"from": "movie_2018", "relation": "has_genre", "to": "genre_Drama"},
                {"from": "genre_Drama",  "relation": "has_genre", "to": "movie_2858"}]}
    """
    history_mid = path_obj.get("history_movie")
    path = path_obj.get("path", [])
    if not path:
        return ""

    history_title = movie_info.get(history_mid, {}).get("title", f"movie {history_mid}")

    # Parse the 2-hop path
    hop1 = path[0] if len(path) > 0 else {}
    hop2 = path[1] if len(path) > 1 else {}

    rel1 = hop1.get("relation", "")
    mid_node = hop1.get("to", "")
    rel2 = hop2.get("relation", "")
    to_node = hop2.get("to", "")

    def node_label(node_str: str) -> str:
        """Convert 'genre_Drama' → 'Drama genre', 'movie_2858' → movie title."""
        if node_str.startswith("genre_"):
            return node_str[6:] + " genre"
        if node_str.startswith("movie_"):
            try:
                mid = int(node_str[6:])
                return movie_info.get(mid, {}).get("title", node_str)
            except ValueError:
                return node_str
        return node_str

    mid_label = node_label(mid_node)
    to_label = node_label(to_node)

    if rel1 == "has_genre" and rel2 == "has_genre":
        return (f"'{history_title}' and '{to_label}' both belong to the {mid_label}.")
    elif rel1 == "co_liked" and rel2 == "co_liked":
        return (f"Users who watched '{history_title}' also frequently watched "
                f"'{mid_label}', and fans of '{mid_label}' often enjoy '{to_label}' as well.")
    else:
        return (f"'{history_title}' connects to '{to_label}' via "
                f"{rel1.replace('_', ' ')} → {mid_label} → {rel2.replace('_', ' ')}.")


def _get_kg_evidence(uid: int, mid: int, kg_paths: dict, movie_info: dict,
                     max_paths: int = KG_MAX_PATHS) -> list[dict]:
    """Return up to max_paths KG-path evidence items for a (user, movie) pair."""
    uid = _normalize_id(uid)
    mid = _normalize_id(mid)
    key = f"{uid}_{mid}"
    paths = kg_paths.get(key, [])
    evidence = []
    for path_obj in paths[:max_paths]:
        text = _format_kg_path_text(path_obj, movie_info)
        if text:
            evidence.append({
                "doc_id": -1,
                "movie_id": mid,
                "text": text,
                "source": "kg_path",
                "dense_score": 1.0,
                "bm25_score": 1.0,
                "score": 1.0,
            })
    return evidence


def _sample_unrelated_kg_evidence(uid: int,
                                  mid: int,
                                  kg_paths: dict,
                                  movie_info: dict,
                                  max_paths: int = KG_MAX_PATHS) -> list[dict]:
    """Sample unrelated KG-path evidence from other recommendation pairs."""
    current_key = _pair_key(uid, mid)
    candidate_keys = [key for key in kg_paths.keys() if key != current_key]
    random.shuffle(candidate_keys)

    sampled = []
    seen_texts = set()
    target_mid = _normalize_id(mid)

    for key in candidate_keys:
        other_uid, other_mid = key.split("_", 1)
        other_mid_int = _normalize_id(other_mid)
        if other_mid_int == target_mid:
            continue

        for ev in _get_kg_evidence(other_uid, other_mid_int, kg_paths, movie_info, max_paths=max_paths):
            text = ev.get("text", "").strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            sampled.append(ev)
            if len(sampled) >= max_paths:
                return sampled

    return sampled


def _perturb_evidence(condition: str,
                      original_evidence: list[dict],
                      evidence_mode: str,
                      *,
                      corpus=None,
                      target_movie_id: int | None = None,
                      uid: int | None = None,
                      mid: int | None = None,
                      kg_paths: dict | None = None,
                      movie_info: dict | None = None) -> list[dict]:
    """Apply an E1-E4 perturbation compatible with the selected evidence mode."""
    from rag.faithfulness import (
        perturb_e1_original, perturb_e2_remove_key,
        perturb_e3_shuffle, perturb_e4_replace_unrelated,
    )

    if condition == "E1":
        return perturb_e1_original(original_evidence)
    if condition == "E2":
        return perturb_e2_remove_key(original_evidence)
    if condition == "E3":
        return perturb_e3_shuffle(original_evidence)
    if condition != "E4":
        raise ValueError(f"Unsupported perturbation condition: {condition}")

    if evidence_mode == "kg_only":
        if uid is None or mid is None or kg_paths is None or movie_info is None:
            raise ValueError("KG-only E4 perturbation requires uid, mid, kg_paths, and movie_info.")
        unrelated = _sample_unrelated_kg_evidence(
            uid, mid, kg_paths, movie_info, max_paths=max(1, len(original_evidence))
        )
        if not unrelated:
            raise ValueError(
                f"Unable to sample unrelated KG evidence for user={uid}, movie={mid}"
            )
        return unrelated

    if corpus is None or target_movie_id is None:
        raise ValueError("Text-grounded E4 perturbation requires corpus and target_movie_id.")
    return perturb_e4_replace_unrelated(original_evidence, corpus, target_movie_id)





def _append_result(path: str, record: dict):
    """Append a single result to a JSONL file (for checkpoint/resume)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_make_serializable(record), ensure_ascii=False) + "\n")


def _attach_reference_evidence(records: list[dict],
                               reference_path: str | None) -> int:
    """Attach reference evidence for evaluation when records do not carry it."""
    reference_map = _load_existing_evidence_map(reference_path) if reference_path else {}
    missing = 0
    attached = 0

    for record in records:
        existing = record.get("reference_evidence")
        if existing:
            attached += 1
            continue

        key = (_normalize_id(record.get("user_id")), _normalize_id(record.get("movie_id")))
        reference = reference_map.get(key)
        if reference:
            record["reference_evidence"] = reference
            attached += 1
        else:
            record["reference_evidence"] = record.get("evidence_used", [])
            if record["reference_evidence"]:
                attached += 1
            else:
                missing += 1

    print(f"Attached reference evidence for {attached}/{len(records)} records "
          f"(missing: {missing})")
    return missing


# ---------------------------------------------------------------------------
# Phase 5.1: Build corpus
# ---------------------------------------------------------------------------

def run_phase_5_1():
    """Build RAG evidence corpus and indices."""
    print("\n" + "#" * 60)
    print("# Phase 5.1: Build RAG Evidence Corpus")
    print("#" * 60)

    from rag.build_corpus import main as build_main
    build_main()


# ---------------------------------------------------------------------------
# Phase 5.2: Generate explanations (with checkpoint/resume)
# ---------------------------------------------------------------------------

def _generate_one(task):
    """Worker function for concurrent LLM generation."""
    generator, req, mode = task
    result = generator.generate_explanation(req, mode=mode)
    return result


def run_phase_5_2(llm_backend: str = "huggingface",
                  model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                  api_url: str = "http://localhost:8000/v1",
                  api_key: str = "not-needed",
                  num_users: int = 0,
                  top_k: int = 10,
                  retrieval_k: int = 8,
                  alpha: float = 0.6,
                  concurrency: int = 16,
                  evidence_mode: str = "hybrid",
                  output_dir: str = "results"):
    """Generate evidence-backed and prompt-only explanations with checkpoint/resume."""
    print("\n" + "#" * 60)
    print("# Phase 5.2: Generate Explanations")
    print("#" * 60)
    print(f"Evidence mode: {evidence_mode}")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from rag.retriever import HybridRetriever, build_query
    from rag.generator import (
        ExplanationGenerator, ExplanationRequest, HuggingFaceLLM,
        APIBackend,
    )

    # Load data
    print("Loading recommendation data...")
    recs = get_top_k_recommendations(k=top_k)
    user_history = get_user_history()
    movie_info = get_movie_info()

    # Load KG structural paths for V4 recommendations (optional enhancement)
    kg_paths = _load_kg_paths(
        required=(evidence_mode in {"hybrid", "kg_only"}),
        context=f"phase 5.2 ({evidence_mode})",
    )
    recs, skipped_no_kg = _filter_recommendations_for_kg_paths(
        recs, kg_paths, evidence_mode=evidence_mode
    )
    if skipped_no_kg:
        print(f"Skipping {skipped_no_kg} recommendation pairs without KG paths in kg_only mode")

    # Select users
    all_users = sorted(_normalize_id(uid) for uid in recs["user_id"].unique())
    if num_users > 0:
        all_users = all_users[:num_users]
    print(f"Will process {len(all_users)} users, top-{top_k} each")

    # Init retriever
    retriever = None
    if evidence_mode != "kg_only":
        print("Loading retriever...")
        retriever_model = "/root/autodl-tmp/models/all-MiniLM-L6-v2"
        if not os.path.exists(retriever_model):
            retriever_model = "all-MiniLM-L6-v2"
        retriever = HybridRetriever(corpus_dir="data/rag", model_name=retriever_model, alpha=alpha)

    # Init LLM
    use_concurrent = (llm_backend == "api")
    if llm_backend == "api":
        llm = APIBackend(base_url=api_url, model=model_name, api_key=api_key)
    else:
        llm = HuggingFaceLLM(model_name=model_name)
        concurrency = 1
    generator = ExplanationGenerator(llm)

    # Checkpoint/resume: load already-done pairs
    os.makedirs(output_dir, exist_ok=True)
    primary_mode = _get_primary_mode_label(evidence_mode)
    primary_path = os.path.join(output_dir, f"explanations_{primary_mode}.jsonl")
    prompt_path = os.path.join(output_dir, "explanations_prompt_only.jsonl")
    primary_done = _load_existing_results(primary_path)
    prompt_done = _load_existing_results(prompt_path)
    primary_reference_map = _load_existing_evidence_map(primary_path)
    print(
        f"Checkpoint: {len(primary_done)} {primary_mode} + "
        f"{len(prompt_done)} prompt-only already done"
    )
    if use_concurrent:
        print(f"Concurrent API requests: {concurrency}")

    total = 0
    skipped = 0
    import threading
    write_lock = threading.Lock()

    def _safe_append(path, record):
        with write_lock:
            _append_result(path, record)

    # 流式处理：以用户为批次，每批先检索再并发生成，避免全量预取检索
    USER_BATCH = max(1, concurrency // top_k + 1)  # 每批用户数，保证并发度足够

    # 总 pair 数（排除已完成）用于进度条
    total_needed = sum(
        1 for uid in all_users
        for rec_row in recs[recs["user_id"] == uid].itertuples(index=False)
        if (uid, _normalize_id(rec_row.movie_id)) not in primary_done
        or (uid, _normalize_id(rec_row.movie_id)) not in prompt_done
    )
    pbar = tqdm(total=total_needed, desc="Pairs")

    def _process_batch_serial(batch_tasks):
        """HuggingFace 后端：串行处理一批任务。"""
        for task_info in batch_tasks:
            uid, mid = task_info["uid"], task_info["mid"]
            minfo = task_info["minfo"]
            if task_info["need_primary"]:
                req = ExplanationRequest(
                    user_id=uid, candidate_movie_id=mid,
                    candidate_title=minfo["title"],
                    candidate_genres=minfo.get("genres", ""),
                    history_titles=task_info["history_titles"],
                    evidence=task_info["evidence"],
                )
                result = generator.generate_explanation(req, mode=primary_mode)
                _append_result(primary_path, {
                    "user_id": result.user_id, "movie_id": result.movie_id,
                    "movie_title": result.movie_title,
                    "explanation": result.explanation,
                    "mode": primary_mode, "evidence_used": result.evidence_used,
                    "reference_evidence": result.evidence_used,
                })
            if task_info["need_prompt"]:
                req = ExplanationRequest(
                    user_id=uid, candidate_movie_id=mid,
                    candidate_title=minfo["title"],
                    candidate_genres=minfo.get("genres", ""),
                    history_titles=task_info["history_titles"],
                    evidence=[],
                )
                result = generator.generate_explanation(req, mode="prompt_only")
                _append_result(prompt_path, {
                    "user_id": result.user_id, "movie_id": result.movie_id,
                    "movie_title": result.movie_title,
                    "explanation": result.explanation,
                    "mode": "prompt_only", "evidence_used": [],
                    "reference_evidence": task_info["evidence"] or [],
                })
            pbar.update(1)

    def _process_batch_concurrent(batch_tasks, executor):
        """API 后端：并发处理一批任务。"""
        futures = {}
        for task_info in batch_tasks:
            uid, mid = task_info["uid"], task_info["mid"]
            minfo = task_info["minfo"]
            if task_info["need_primary"]:
                req = ExplanationRequest(
                    user_id=uid, candidate_movie_id=mid,
                    candidate_title=minfo["title"],
                    candidate_genres=minfo.get("genres", ""),
                    history_titles=task_info["history_titles"],
                    evidence=task_info["evidence"],
                )
                fut = executor.submit(_generate_one, (generator, req, primary_mode))
                futures[fut] = (primary_mode, task_info)
            if task_info["need_prompt"]:
                req = ExplanationRequest(
                    user_id=uid, candidate_movie_id=mid,
                    candidate_title=minfo["title"],
                    candidate_genres=minfo.get("genres", ""),
                    history_titles=task_info["history_titles"],
                    evidence=[],
                )
                fut = executor.submit(_generate_one, (generator, req, "prompt_only"))
                futures[fut] = ("prompt_only", task_info)

        completed_pairs = set()
        for fut in as_completed(futures):
            mode, task_info = futures[fut]
            try:
                result = fut.result()
                path = primary_path if mode == primary_mode else prompt_path
                _safe_append(path, {
                    "user_id": result.user_id,
                    "movie_id": result.movie_id,
                    "movie_title": result.movie_title,
                    "explanation": result.explanation,
                    "mode": mode,
                    "evidence_used": result.evidence_used if mode == primary_mode else [],
                    "reference_evidence": (result.evidence_used if mode == primary_mode
                                           else (task_info["evidence"] or [])),
                })
            except Exception as e:
                print(f"\nError ({mode}) user={task_info['uid']} "
                      f"movie={task_info['mid']}: {e}")
            pair_key = (task_info["uid"], task_info["mid"])
            if pair_key not in completed_pairs:
                completed_pairs.add(pair_key)
                pbar.update(1)

    executor = ThreadPoolExecutor(max_workers=concurrency) if use_concurrent else None

    # 以用户批次流式处理：检索 + 生成交替进行
    for batch_start in range(0, len(all_users), USER_BATCH):
        batch_users = all_users[batch_start: batch_start + USER_BATCH]
        batch_tasks = []

        for uid in batch_users:
            uid = _normalize_id(uid)
            user_recs = recs[recs["user_id"] == uid]
            history_mids = user_history.get(uid, [])
            history_titles = [movie_info[m]["title"]
                              for m in history_mids if m in movie_info]

            for rec_row in user_recs.itertuples(index=False):
                mid = _normalize_id(rec_row.movie_id)
                minfo = movie_info.get(mid, {"title": f"Movie {mid}", "genres": ""})
                pair = (uid, mid)

                need_primary = pair not in primary_done
                need_prompt = pair not in prompt_done

                if not need_primary:
                    skipped += 1
                if not need_primary and not need_prompt:
                    total += 1
                    continue

                # Share one reference evidence set across primary mode + prompt-only.
                evidence = primary_reference_map.get(pair)
                if evidence is None and (need_primary or need_prompt):
                    retrieved_evidence = []
                    if evidence_mode != "kg_only":
                        query = build_query(
                            candidate_title=minfo["title"],
                            candidate_genres=minfo.get("genres", ""),
                            history_titles=history_titles,
                        )
                        retrieved_evidence = retriever.retrieve(
                            query=query, top_k=retrieval_k,
                            candidate_movie_id=mid,
                            history_movie_ids=history_mids[:5],
                        )

                    kg_ev = []
                    if evidence_mode != "retrieval_only":
                        kg_ev = _get_kg_evidence(uid, mid, kg_paths, movie_info, max_paths=KG_MAX_PATHS)

                    evidence = _build_evidence_bundle(
                        retrieved_evidence,
                        kg_ev,
                        evidence_mode=evidence_mode,
                    )
                    if evidence_mode == "kg_only" and not evidence:
                        print(f"[WARN] Skipping user={uid}, movie={mid} because no KG evidence was found")
                        total += 1
                        continue

                batch_tasks.append({
                    "uid": uid, "mid": mid, "minfo": minfo,
                    "history_titles": history_titles,
                    "evidence": evidence,
                    "need_primary": need_primary,
                    "need_prompt": need_prompt,
                })
                total += 1

        if not batch_tasks:
            continue

        if use_concurrent:
            _process_batch_concurrent(batch_tasks, executor)
        else:
            _process_batch_serial(batch_tasks)

    if executor:
        executor.shutdown(wait=True)

    pbar.close()
    print(
        f"\nDone! Processed {total} pairs "
        f"({skipped} {primary_mode} records skipped from checkpoint)"
    )


# ---------------------------------------------------------------------------
# Phase 5.3: Perturbation experiments (E1-E4)
# ---------------------------------------------------------------------------

def run_phase_5_3(llm_backend: str = "huggingface",
                  model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                  api_url: str = "http://localhost:8000/v1",
                  api_key: str = "not-needed",
                  num_samples: int = 200,
                  top_k: int = 10,
                  retrieval_k: int = 8,
                  evidence_mode: str = "hybrid",
                  output_dir: str = "results"):
    """Run E1-E4 perturbation experiments on a sample of recommendations."""
    print("\n" + "#" * 60)
    print("# Phase 5.3: Perturbation Experiments (E1-E4)")
    print("#" * 60)
    print(f"Evidence mode: {evidence_mode}")

    from rag.retriever import HybridRetriever, build_query
    from rag.generator import (
        ExplanationGenerator, ExplanationRequest, HuggingFaceLLM,
        APIBackend,
    )
    # Load data
    recs = get_top_k_recommendations(k=top_k)
    user_history = get_user_history()
    movie_info = get_movie_info()
    kg_paths = _load_kg_paths(
        required=(evidence_mode in {"hybrid", "kg_only"}),
        context=f"phase 5.3 ({evidence_mode})",
    )
    recs, skipped_no_kg = _filter_recommendations_for_kg_paths(
        recs, kg_paths, evidence_mode=evidence_mode
    )
    if skipped_no_kg:
        print(f"Skipping {skipped_no_kg} recommendation pairs without KG paths in kg_only mode")

    # Sample user-movie pairs
    all_pairs = list(recs[["user_id", "movie_id"]].itertuples(index=False))
    if num_samples > 0 and len(all_pairs) > num_samples:
        random.seed(42)
        all_pairs = random.sample(all_pairs, num_samples)

    # Init components
    retriever = None
    corpus = None
    if evidence_mode != "kg_only":
        retriever_model = "/root/autodl-tmp/models/all-MiniLM-L6-v2"
        if not os.path.exists(retriever_model):
            retriever_model = "all-MiniLM-L6-v2"
        retriever = HybridRetriever(corpus_dir="data/rag", model_name=retriever_model)
        corpus = retriever.corpus

    if llm_backend == "api":
        llm = APIBackend(base_url=api_url, model=model_name, api_key=api_key)
    else:
        llm = HuggingFaceLLM(model_name=model_name)
    generator = ExplanationGenerator(llm)
    primary_mode = _get_primary_mode_label(evidence_mode)

    # Checkpoint/resume
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "perturbation_results.jsonl")
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((_normalize_id(r["user_id"]), _normalize_id(r["movie_id"]), r["condition"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"Checkpoint: {len(done)} results already done")

    conditions = ["E1", "E2", "E3", "E4"]

    for uid, mid in tqdm(all_pairs, desc="Perturbation"):
        uid = _normalize_id(uid)
        mid = _normalize_id(mid)
        history_mids = user_history.get(uid, [])
        history_titles = [movie_info[m]["title"]
                          for m in history_mids if m in movie_info]
        minfo = movie_info.get(mid, {"title": f"Movie {mid}", "genres": ""})

        retrieved_evidence = []
        if evidence_mode != "kg_only":
            query = build_query(
                candidate_title=minfo["title"],
                candidate_genres=minfo.get("genres", ""),
                history_titles=history_titles,
            )
            retrieved_evidence = retriever.retrieve(
                query=query, top_k=retrieval_k,
                candidate_movie_id=mid, history_movie_ids=history_mids[:5],
            )

        kg_ev = []
        if evidence_mode != "retrieval_only":
            kg_ev = _get_kg_evidence(uid, mid, kg_paths, movie_info, max_paths=KG_MAX_PATHS)

        original_evidence = _build_evidence_bundle(
            retrieved_evidence,
            kg_ev,
            evidence_mode=evidence_mode,
        )
        if evidence_mode == "kg_only" and not original_evidence:
            print(f"[WARN] Skipping user={uid}, movie={mid} because no KG evidence was found")
            continue

        for condition in conditions:
            if (uid, mid, condition) in done:
                continue

            evidence = _perturb_evidence(
                condition,
                original_evidence,
                evidence_mode,
                corpus=corpus,
                target_movie_id=mid,
                uid=uid,
                mid=mid,
                kg_paths=kg_paths,
                movie_info=movie_info,
            )

            req = ExplanationRequest(
                user_id=uid, candidate_movie_id=mid,
                candidate_title=minfo["title"],
                candidate_genres=minfo.get("genres", ""),
                history_titles=history_titles,
                evidence=evidence,
            )
            result = generator.generate_explanation(req, mode=primary_mode)

            _append_result(output_path, {
                "user_id": uid, "movie_id": mid,
                "condition": condition,
                "mode": primary_mode,
                "explanation": result.explanation,
                "evidence_used": evidence,
            })

    print(f"Perturbation results saved -> {output_path}")


# ---------------------------------------------------------------------------
# Phase 5.4: Evaluate faithfulness
# ---------------------------------------------------------------------------

def run_phase_5_4(output_dir: str = "results"):
    """Evaluate faithfulness metrics for all generated explanations."""
    print("\n" + "#" * 60)
    print("# Phase 5.4: Faithfulness Evaluation")
    print("#" * 60)

    from rag.faithfulness import (
        evaluate_faithfulness, aggregate_results,
        save_faithfulness_results,
    )

    reference_path = _pick_reference_explanation_path(output_dir)

    # Evaluate all explanation modes present in this output directory.
    for mode, path in _iter_explanation_files(output_dir):
        print(f"\nEvaluating {mode} explanations...")
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["condition"] = mode.upper()
                records.append(r)

        _attach_reference_evidence(records, reference_path=reference_path)
        results = evaluate_faithfulness(records)
        summary = aggregate_results(results)
        save_faithfulness_results(
            results, summary,
            output_dir=os.path.join(output_dir, f"faithfulness_{mode}"))

    # Evaluate perturbation experiments
    perturb_path = os.path.join(output_dir, "perturbation_results.jsonl")
    if os.path.exists(perturb_path):
        print("\nEvaluating perturbation experiments (E1-E4)...")
        records = []
        with open(perturb_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        _attach_reference_evidence(records, reference_path=reference_path)
        results = evaluate_faithfulness(records)
        summary = aggregate_results(results)
        save_faithfulness_results(
            results, summary,
            output_dir=os.path.join(output_dir, "faithfulness_perturbation"))
    else:
        print(f"Skipping perturbation eval ({perturb_path} not found)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG Explanation Pipeline (Stage 3)")
    parser.add_argument("--phase", type=str, default=None,
                        help="Run specific phase (5.1/5.2/5.3/5.4)")
    parser.add_argument("--llm-backend", type=str, default="huggingface",
                        choices=["huggingface", "api"],
                        help="LLM backend type")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name/path")
    parser.add_argument("--api-url", type=str,
                        default="http://localhost:8000/v1",
                        help="API base URL (for api backend)")
    parser.add_argument("--api-key", type=str, default="not-needed",
                        help="API key (for api backend)")
    parser.add_argument("--num-users", type=int, default=0,
                        help="Limit number of users (0 = all)")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of samples for perturbation experiments")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K recommendations to explain")
    parser.add_argument("--retrieval-k", type=int, default=8,
                        help="Number of evidence chunks to retrieve")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Dense retrieval weight (1-alpha for BM25)")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Number of concurrent API requests (api backend only)")
    parser.add_argument("--evidence-mode", type=str, default="hybrid",
                        choices=["hybrid", "kg_only", "retrieval_only"],
                        help="Evidence source for the primary explanation mode")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for phase 5.3/5.4 results")
    args = parser.parse_args()

    if args.phase == "5.1":
        run_phase_5_1()
    elif args.phase == "5.2":
        run_phase_5_2(
            llm_backend=args.llm_backend, model_name=args.model,
            api_url=args.api_url, api_key=args.api_key,
            num_users=args.num_users, top_k=args.top_k,
            retrieval_k=args.retrieval_k, alpha=args.alpha,
            concurrency=args.concurrency,
            evidence_mode=args.evidence_mode,
            output_dir=args.output_dir,
        )
    elif args.phase == "5.3":
        run_phase_5_3(
            llm_backend=args.llm_backend, model_name=args.model,
            api_url=args.api_url, api_key=args.api_key,
            num_samples=args.num_samples, top_k=args.top_k,
            retrieval_k=args.retrieval_k,
            evidence_mode=args.evidence_mode,
            output_dir=args.output_dir,
        )
    elif args.phase == "5.4":
        run_phase_5_4(output_dir=args.output_dir)
    else:
        run_phase_5_1()
        run_phase_5_2(
            llm_backend=args.llm_backend, model_name=args.model,
            api_url=args.api_url, api_key=args.api_key,
            num_users=args.num_users, top_k=args.top_k,
            retrieval_k=args.retrieval_k, alpha=args.alpha,
            concurrency=args.concurrency,
            evidence_mode=args.evidence_mode,
            output_dir=args.output_dir,
        )
        run_phase_5_3(
            llm_backend=args.llm_backend, model_name=args.model,
            api_url=args.api_url, api_key=args.api_key,
            num_samples=args.num_samples, top_k=args.top_k,
            retrieval_k=args.retrieval_k,
            evidence_mode=args.evidence_mode,
            output_dir=args.output_dir,
        )
        run_phase_5_4(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
