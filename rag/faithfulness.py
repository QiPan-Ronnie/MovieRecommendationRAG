"""
Faithfulness evaluation for RAG-generated explanations (RQ3).

Metrics:
  1. Evidence Overlap Score: token/phrase overlap between explanation and evidence
  2. Semantic Similarity: embedding cosine similarity between explanation and evidence
  3. ROUGE-L: longest common subsequence recall
  4. BERTScore: contextualized embedding similarity

Perturbation experiments (E1-E4):
  E1: Use original evidence (baseline)
  E2: Remove key evidence sentences (test evidence dependency)
  E3: Shuffle evidence order (test robustness)
  E4: Replace with unrelated evidence (test faithfulness)
"""
import json
import os
import random
import time
import datetime
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Evidence normalization helpers
# ---------------------------------------------------------------------------


def _extract_evidence_texts(record: dict) -> list[str]:
    """
    Get canonical evidence texts for evaluation.

    Preference order:
      1. reference_evidence: fixed evidence paired with the sample for eval.
      2. evidence_used: evidence actually passed to generation.

    Canonicalizing order prevents E3 (shuffle) from changing the metric
    reference itself.
    """
    raw_evidence = record.get("reference_evidence")
    if not raw_evidence:
        raw_evidence = record.get("evidence_used", [])

    normalized = []
    for idx, ev in enumerate(raw_evidence):
        if isinstance(ev, dict):
            text = str(ev.get("text", "")).strip()
            doc_id = ev.get("doc_id")
        else:
            text = str(ev).strip()
            doc_id = None

        if not text:
            continue

        if isinstance(doc_id, (int, np.integer)):
            sort_key = (0, int(doc_id), idx)
        else:
            sort_key = (1, idx, idx)
        normalized.append((sort_key, text))

    normalized.sort(key=lambda item: item[0])
    texts = [text for _, text in normalized]
    return texts if texts else [""]


# ---------------------------------------------------------------------------
# Original metric implementations (kept for backward compat)
# ---------------------------------------------------------------------------

def evidence_overlap_score(explanation: str, evidence_texts: list[str]) -> float:
    if not explanation or not evidence_texts:
        return 0.0
    exp_tokens = set(explanation.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                  "at", "to", "for", "of", "and", "or", "but", "it", "this",
                  "that", "with", "by", "from", "as", "be", "has", "have",
                  "had", "do", "does", "did", "will", "would", "can", "could",
                  "not", "no", "so", "if", "its", "also", "than", "then"}
    exp_tokens -= stop_words
    if not exp_tokens:
        return 0.0
    evidence_tokens = set()
    for ev in evidence_texts:
        evidence_tokens.update(ev.lower().split())
    evidence_tokens -= stop_words
    overlap = exp_tokens & evidence_tokens
    return len(overlap) / len(exp_tokens)


def rouge_l(explanation: str, evidence_texts: list[str]) -> float:
    if not explanation or not evidence_texts:
        return 0.0
    reference = " ".join(evidence_texts).lower().split()
    hypothesis = explanation.lower().split()
    if not reference or not hypothesis:
        return 0.0
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    return lcs_len / n if n > 0 else 0.0


def semantic_similarity(explanation: str, evidence_texts: list[str],
                        model=None) -> float:
    """Kept for backward compatibility; main flow uses batched version."""
    if not explanation or not evidence_texts:
        return 0.0
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    all_texts = [explanation] + evidence_texts
    embeddings = model.encode(all_texts, normalize_embeddings=True)
    exp_emb = embeddings[0]
    ev_embs = embeddings[1:]
    sims = [float(np.dot(exp_emb, ev_emb)) for ev_emb in ev_embs]
    return float(np.mean(sims))


def compute_bertscore(explanations: list[str],
                      evidence_list: list[list[str]]) -> list[float]:
    """Kept for backward compatibility; main flow uses batched version."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("WARNING: bert-score not installed.", flush=True)
        return [0.0] * len(explanations)
    references = [" ".join(evs) for evs in evidence_list]
    local_roberta = "/root/autodl-tmp/models/roberta-large"
    if os.path.isdir(local_roberta):
        _, _, f1 = bert_score_fn(explanations, references,
                                 model_type=local_roberta, num_layers=17,
                                 lang=None, device="cuda", verbose=False,
                                 rescale_with_baseline=False)
    else:
        _, _, f1 = bert_score_fn(explanations, references, lang="en",
                                 verbose=False, rescale_with_baseline=True)
    return f1.tolist()


# ---------------------------------------------------------------------------
# Batched / optimized implementations (Phase 5.4 v3)
# ---------------------------------------------------------------------------

def compute_bertscore_batched(explanations: list[str],
                              evidence_list: list[list[str]],
                              batch_size: int = 256) -> list[float]:
    """BERTScore in mini-batches with one reusable scorer + live ETA logs."""
    try:
        from bert_score import BERTScorer
    except ImportError:
        print("  WARNING: bert-score not installed. Returning zeros.", flush=True)
        return [0.0] * len(explanations)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    local_roberta = "/root/autodl-tmp/models/roberta-large"
    use_local = os.path.isdir(local_roberta)
    model_label = local_roberta if use_local else "roberta-large"
    print(f"  BERTScore device: {device}", flush=True)
    print(f"  BERTScore model: {model_label}", flush=True)

    scorer_kwargs = {
        "device": device,
    }
    if use_local:
        scorer_kwargs.update({
            "model_type": local_roberta,
            "num_layers": 17,
            "lang": None,
            "rescale_with_baseline": False,
        })
    else:
        scorer_kwargs.update({
            "lang": "en",
            "rescale_with_baseline": True,
        })

    scorer = BERTScorer(**scorer_kwargs)

    references = [" ".join(evs) for evs in evidence_list]
    n = len(explanations)
    all_f1: list[float] = []
    n_batches = (n + batch_size - 1) // batch_size
    t_start = time.time()
    running_sum = 0.0
    inner_batch_size = 64 if device == "cuda" else 256

    with tqdm(total=n_batches, desc="  BERTScore", unit="batch", dynamic_ncols=True) as pbar:
        for batch_idx in range(n_batches):
            lo = batch_idx * batch_size
            hi = min(lo + batch_size, n)
            _, _, f1 = scorer.score(
                explanations[lo:hi],
                references[lo:hi],
                batch_size=inner_batch_size,
                verbose=False,
            )

            batch_f1 = f1.tolist()
            all_f1.extend(batch_f1)
            running_sum += sum(batch_f1)
            records_done = hi
            avg_f1 = running_sum / records_done
            elapsed = time.time() - t_start
            rate = records_done / elapsed if elapsed > 0 else 1e-9
            remaining = (n - records_done) / rate
            pbar.set_postfix(avg_f1=f"{avg_f1:.4f}", done=f"{records_done}/{n}")
            pbar.update(1)

            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                eta_str = str(datetime.timedelta(seconds=int(remaining)))
                elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
                print(f"  [??] {records_done}/{n} | avg_F1={avg_f1:.4f} | "
                      f"??: {elapsed_str} | ETA: ~{eta_str}  [{ts}]", flush=True)

    return all_f1


def compute_semantic_similarity_batched(explanations: list[str],
                                        evidence_list: list[list[str]],
                                        model=None) -> list[float]:
    """Bulk encode all texts, then per-record dot-product. Very efficient."""
    n = len(explanations)
    if n == 0:
        return []
    if model is None:
        print("  ?? all-MiniLM-L6-v2 ...", flush=True)
        from sentence_transformers import SentenceTransformer
        local_minilm = "/root/autodl-tmp/models/all-MiniLM-L6-v2"
        try:
            import torch
            st_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            st_device = "cpu"
        if os.path.isdir(local_minilm):
            print(f"  ??????: {local_minilm} (device={st_device})", flush=True)
            model = SentenceTransformer(local_minilm, local_files_only=True, device=st_device)
        else:
            print(f"  ?????????? HF Hub (device={st_device})", flush=True)
            model = SentenceTransformer("all-MiniLM-L6-v2", device=st_device)
        print("  ??????.", flush=True)


    print(f"  编码 {n} 条 explanations ...", flush=True)
    exp_embs = model.encode(explanations, normalize_embeddings=True,
                            show_progress_bar=True, batch_size=256)

    flat_ev: list[str] = []
    ev_slices: list[tuple] = []
    for ev_texts in evidence_list:
        lo = len(flat_ev)
        flat_ev.extend(ev_texts if ev_texts else [""])
        ev_slices.append((lo, len(flat_ev)))

    print(f"  编码 {len(flat_ev)} 条 evidence texts ...", flush=True)
    ev_embs = model.encode(flat_ev, normalize_embeddings=True,
                           show_progress_bar=True, batch_size=256)

    print("  计算余弦相似度 ...", flush=True)
    sims: list[float] = []
    for i, (lo, hi) in enumerate(ev_slices):
        per = np.dot(ev_embs[lo:hi], exp_embs[i])
        sims.append(float(np.mean(per)))
    return sims


# ---------------------------------------------------------------------------
# Perturbation functions for E1-E4
# ---------------------------------------------------------------------------

def perturb_e1_original(evidence: list[dict]) -> list[dict]:
    return evidence

def perturb_e2_remove_key(evidence: list[dict], remove_ratio: float = 0.5) -> list[dict]:
    if not evidence:
        return evidence
    sorted_ev = sorted(evidence, key=lambda x: x.get("score", 0), reverse=True)
    n_remove = max(1, int(len(sorted_ev) * remove_ratio))
    return sorted_ev[n_remove:]

def perturb_e3_shuffle(evidence: list[dict]) -> list[dict]:
    shuffled = list(evidence)
    random.shuffle(shuffled)
    return shuffled

def perturb_e4_replace_unrelated(evidence: list[dict], corpus: list[dict],
                                  candidate_movie_id: int) -> list[dict]:
    unrelated = [doc for doc in corpus if doc.get("movie_id") != candidate_movie_id]
    if not unrelated:
        return evidence
    n = len(evidence)
    sampled = random.sample(unrelated, min(n, len(unrelated)))
    return [{"doc_id": s.get("doc_id", -1), "movie_id": s["movie_id"],
             "text": s["text"], "source": s.get("source", "unrelated"),
             "score": 0.0, "dense_score": 0.0, "bm25_score": 0.0}
            for s in sampled]

PERTURBATION_FNS = {
    "E1": perturb_e1_original,
    "E2": perturb_e2_remove_key,
    "E3": perturb_e3_shuffle,
}


# ---------------------------------------------------------------------------
# FaithfulnessResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class FaithfulnessResult:
    user_id: int
    movie_id: int
    condition: str
    explanation: str
    evidence_overlap: float
    rouge_l: float
    semantic_sim: float
    bert_score_f1: float = 0.0


def _fmt_dur(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_faithfulness(explanations: list[dict],
                          st_model=None) -> list[FaithfulnessResult]:
    """
    4-phase pipeline with real-time progress:
      [1/4] BERTScore       - batched, ETA every 10 batches
      [2/4] Semantic Sim    - bulk encode, dot-product
      [3/4] Overlap+ROUGE-L - tqdm loop
      [4/4] Assemble + print summary table
    """
    n = len(explanations)
    SEP = "  " + "-" * 52
    start_ts = datetime.datetime.now()

    print("", flush=True)
    print("  " + "=" * 52, flush=True)
    print(f"  [Phase 5.4] 开始: {n} 条记录  | {start_ts.strftime('%H:%M:%S')}", flush=True)
    print("  " + "=" * 52, flush=True)
    print("", flush=True)

    all_exps = [e["explanation"] for e in explanations]
    all_evs: list[list[str]] = []
    for e in explanations:
        all_evs.append(_extract_evidence_texts(e))

    # [1/4] BERTScore
    print(SEP, flush=True)
    print("  [1/4] BERTScore  (roberta-large, device=cpu)", flush=True)
    print(SEP, flush=True)
    t0 = time.time()
    bert_scores = compute_bertscore_batched(all_exps, all_evs, batch_size=256)
    t1 = time.time()
    mean_bert = float(np.mean(bert_scores)) if bert_scores else 0.0
    print(f"  [完成] 耗时 {_fmt_dur(t1-t0)} | 均值 F1 = {mean_bert:.4f}", flush=True)
    print("", flush=True)

    # [2/4] Semantic Similarity
    print(SEP, flush=True)
    print("  [2/4] Semantic Similarity  (all-MiniLM-L6-v2)", flush=True)
    print(SEP, flush=True)
    t0 = time.time()
    sem_scores = compute_semantic_similarity_batched(all_exps, all_evs, model=st_model)
    t1 = time.time()
    mean_sem = float(np.mean(sem_scores)) if sem_scores else 0.0
    print(f"  [完成] 耗时 {_fmt_dur(t1-t0)} | 均值 = {mean_sem:.4f}", flush=True)
    print("", flush=True)

    # [3/4] Overlap + ROUGE-L
    print(SEP, flush=True)
    print("  [3/4] Evidence Overlap + ROUGE-L", flush=True)
    print(SEP, flush=True)
    t0 = time.time()
    overlaps: list[float] = []
    rouges: list[float] = []
    for record in tqdm(explanations, desc="  Overlap+ROUGE", unit="rec", dynamic_ncols=True):
        exp_t = record["explanation"]
        ev_t = _extract_evidence_texts(record)
        overlaps.append(evidence_overlap_score(exp_t, ev_t))
        rouges.append(rouge_l(exp_t, ev_t))
    t1 = time.time()
    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    mean_rouge = float(np.mean(rouges)) if rouges else 0.0
    print(f"  [完成] 耗时 {_fmt_dur(t1-t0)} | Overlap={mean_overlap:.4f} | ROUGE-L={mean_rouge:.4f}", flush=True)
    print("", flush=True)

    # [4/4] Assemble
    print(SEP, flush=True)
    print("  [4/4] 汇总结果", flush=True)
    print(SEP, flush=True)
    results: list[FaithfulnessResult] = []
    for i, record in enumerate(explanations):
        results.append(FaithfulnessResult(
            user_id=record["user_id"],
            movie_id=record["movie_id"],
            condition=record.get("condition", "E1"),
            explanation=all_exps[i],
            evidence_overlap=round(overlaps[i], 4),
            rouge_l=round(rouges[i], 4),
            semantic_sim=round(sem_scores[i], 4),
            bert_score_f1=round(bert_scores[i], 4),
        ))

    total = (datetime.datetime.now() - start_ts).total_seconds()
    print("", flush=True)
    print(f"  {'指标':<14} {'Overlap':>10} {'ROUGE-L':>10} {'Sem.Sim':>10} {'BERTScore':>10}", flush=True)
    print("  " + "-" * 46, flush=True)
    print(f"  {'(mean)':<14} {mean_overlap:>10.4f} {mean_rouge:>10.4f} {mean_sem:>10.4f} {mean_bert:>10.4f}", flush=True)
    print("", flush=True)
    print(f"  总耗时: {_fmt_dur(total)} | 完成: {datetime.datetime.now().strftime('%H:%M:%S')}", flush=True)
    print("", flush=True)
    return results


def aggregate_results(results: list[FaithfulnessResult]) -> dict:
    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r.condition].append(r)
    summary = {}
    for condition, items in sorted(by_condition.items()):
        summary[condition] = {
            "count": len(items),
            "evidence_overlap": round(np.mean([i.evidence_overlap for i in items]), 4),
            "rouge_l": round(np.mean([i.rouge_l for i in items]), 4),
            "semantic_sim": round(np.mean([i.semantic_sim for i in items]), 4),
            "bert_score_f1": round(np.mean([i.bert_score_f1 for i in items]), 4),
        }
    return summary


def save_faithfulness_results(results: list[FaithfulnessResult],
                               summary: dict, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    detail_path = os.path.join(output_dir, "faithfulness_detailed.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "user_id": r.user_id, "movie_id": r.movie_id,
                "condition": r.condition, "explanation": r.explanation,
                "evidence_overlap": r.evidence_overlap, "rouge_l": r.rouge_l,
                "semantic_sim": r.semantic_sim, "bert_score_f1": r.bert_score_f1,
            }, ensure_ascii=False) + "\n")
    summary_path = os.path.join(output_dir, "faithfulness_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> 详细结果: {detail_path}", flush=True)
    print(f"  -> 汇总文件: {summary_path}", flush=True)
    print(f"\n  {'条件':<12} {'Overlap':>10} {'ROUGE-L':>10} {'Sem.Sim':>10} {'BERTScore':>10}", flush=True)
    print("  " + "-" * 54, flush=True)
    for cond, metrics in sorted(summary.items()):
        print(f"  {cond:<12} {metrics['evidence_overlap']:>10.4f} "
              f"{metrics['rouge_l']:>10.4f} "
              f"{metrics['semantic_sim']:>10.4f} "
              f"{metrics['bert_score_f1']:>10.4f}", flush=True)
