"""Paired significance testing for Phase 5 explanation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_RESULTS_ROOT = Path("/root/autodl-tmp/MovieRecommendationRAG/results")
METRICS = [
    "evidence_overlap",
    "rouge_l",
    "semantic_sim",
    "bert_score_f1",
]


def normalize_id(value) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            raise ValueError("ID value is NaN")
        if not float(value).is_integer():
            raise ValueError(f"Expected integral ID, got {value!r}")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty ID value")
        parsed = float(text)
        if not parsed.is_integer():
            raise ValueError(f"Expected integral ID, got {value!r}")
        return int(parsed)
    return int(value)


def build_pair_key(record: dict) -> tuple[int, int]:
    return normalize_id(record["user_id"]), normalize_id(record["movie_id"])


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def filter_records(records: Iterable[dict], condition: str | None = None) -> list[dict]:
    if condition is None:
        return list(records)
    return [record for record in records if record.get("condition") == condition]


def align_records_on_pairs(left_records: list[dict],
                           right_records: list[dict],
                           metric_name: str) -> dict:
    left_map = {build_pair_key(record): float(record[metric_name]) for record in left_records}
    right_map = {build_pair_key(record): float(record[metric_name]) for record in right_records}
    pair_keys = sorted(set(left_map) & set(right_map))
    return {
        "pair_keys": pair_keys,
        "left_values": [left_map[key] for key in pair_keys],
        "right_values": [right_map[key] for key in pair_keys],
    }


def bootstrap_mean_difference(diff_values: list[float],
                              num_rounds: int = 2000,
                              seed: int = 42,
                              batch_size: int = 50) -> dict:
    diffs = np.asarray(diff_values, dtype=float)
    if diffs.size == 0:
        raise ValueError("Cannot bootstrap an empty set of differences")

    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, num_rounds, batch_size):
        current_batch = min(batch_size, num_rounds - start)
        indices = rng.integers(0, diffs.size, size=(current_batch, diffs.size))
        means.extend(diffs[indices].mean(axis=1).tolist())

    mean_difference = float(diffs.mean())
    ci_low, ci_high = np.quantile(np.asarray(means), [0.025, 0.975]).tolist()
    return {
        "mean_difference": mean_difference,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def summarize_metric_comparison(left_records: list[dict],
                                right_records: list[dict],
                                metric_name: str,
                                left_label: str,
                                right_label: str,
                                bootstrap_rounds: int = 2000,
                                seed: int = 42) -> dict:
    aligned = align_records_on_pairs(left_records, right_records, metric_name)
    left_values = np.asarray(aligned["left_values"], dtype=float)
    right_values = np.asarray(aligned["right_values"], dtype=float)
    pair_count = int(left_values.size)

    if pair_count == 0:
        raise ValueError(
            f"No overlapping user/movie pairs for {left_label} vs {right_label} "
            f"on metric {metric_name}"
        )

    diffs = left_values - right_values
    bootstrap = bootstrap_mean_difference(
        diffs.tolist(),
        num_rounds=bootstrap_rounds,
        seed=seed,
    )

    if pair_count >= 2:
        t_stat, p_value = stats.ttest_rel(left_values, right_values)
        t_stat = float(t_stat)
        p_value = float(p_value)
    else:
        t_stat = None
        p_value = None

    return {
        "left_label": left_label,
        "right_label": right_label,
        "metric": metric_name,
        "pair_count": pair_count,
        "left_mean": float(left_values.mean()),
        "right_mean": float(right_values.mean()),
        "mean_difference": bootstrap["mean_difference"],
        "ci_low": bootstrap["ci_low"],
        "ci_high": bootstrap["ci_high"],
        "t_statistic": t_stat,
        "p_value": p_value,
    }


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the candidate paths exists:\n" + "\n".join(map(str, paths)))


def resolve_default_datasets(results_root: Path) -> dict[str, Path]:
    return {
        "hybrid": first_existing_path(
            results_root / "phase5_with_recommendation_Hybrid_bertscore_unified" / "faithfulness_rag" / "faithfulness_detailed.jsonl",
            results_root / "phase5_with_recommendations_v4&KG_Path" / "faithfulness_rag" / "faithfulness_detailed.jsonl",
        ),
        "hybrid_prompt": first_existing_path(
            results_root / "phase5_with_recommendation_Hybrid_bertscore_unified" / "faithfulness_prompt_only" / "faithfulness_detailed.jsonl",
            results_root / "phase5_with_recommendations_v4&KG_Path" / "faithfulness_prompt_only" / "faithfulness_detailed.jsonl",
        ),
        "retrieval_only": first_existing_path(
            results_root / "phase5_with_recommendation_Retrieval_Only_bertscore_unified" / "faithfulness_rag" / "faithfulness_detailed.jsonl",
            results_root / "backup_pre_kg_fix_20260403_152531" / "phase5_with_recommendations_v4" / "faithfulness_rag" / "faithfulness_detailed.jsonl",
        ),
        "retrieval_prompt": first_existing_path(
            results_root / "phase5_with_recommendation_Retrieval_Only_bertscore_unified" / "faithfulness_prompt_only" / "faithfulness_detailed.jsonl",
            results_root / "backup_pre_kg_fix_20260403_152531" / "phase5_with_recommendations_v4" / "faithfulness_prompt_only" / "faithfulness_detailed.jsonl",
        ),
        "kg_only": first_existing_path(
            results_root / "phase5_with_recommendation_KG_Only_bertscore_unified" / "faithfulness_kg_only" / "faithfulness_detailed.jsonl",
            results_root / "phase5_with_recommendation_KG_Only" / "faithfulness_kg_only" / "faithfulness_detailed.jsonl",
        ),
        "kg_prompt": first_existing_path(
            results_root / "phase5_with_recommendation_KG_Only_bertscore_unified" / "faithfulness_prompt_only" / "faithfulness_detailed.jsonl",
            results_root / "phase5_with_recommendation_KG_Only" / "faithfulness_prompt_only" / "faithfulness_detailed.jsonl",
        ),
        "hybrid_perturb": first_existing_path(
            results_root / "phase5_with_recommendation_Hybrid_p500" / "faithfulness_perturbation" / "faithfulness_detailed.jsonl",
            results_root / "phase5_with_recommendations_v4&KG_Path" / "faithfulness_perturbation" / "faithfulness_detailed.jsonl",
        ),
        "retrieval_perturb": first_existing_path(
            results_root / "phase5_with_recommendation_Retrieval_Only_p500" / "faithfulness_perturbation" / "faithfulness_detailed.jsonl",
            results_root / "backup_pre_kg_fix_20260403_152531" / "phase5_with_recommendations_v4" / "faithfulness_perturbation" / "faithfulness_detailed.jsonl",
        ),
        "kg_perturb": first_existing_path(
            results_root / "phase5_with_recommendation_KG_Only_p500" / "faithfulness_perturbation" / "faithfulness_detailed.jsonl",
            results_root / "phase5_with_recommendation_KG_Only" / "faithfulness_perturbation" / "faithfulness_detailed.jsonl",
        ),
    }


def run_main_comparisons(dataset_paths: dict[str, Path],
                         bootstrap_rounds: int,
                         seed: int) -> pd.DataFrame:
    comparisons = [
        ("hybrid", "retrieval_only"),
        ("hybrid", "kg_only"),
        ("hybrid", "hybrid_prompt"),
        ("retrieval_only", "retrieval_prompt"),
        ("kg_only", "kg_prompt"),
        ("kg_only", "retrieval_only"),
    ]

    loaded = {
        label: load_jsonl(path)
        for label, path in dataset_paths.items()
        if label in {"hybrid", "retrieval_only", "kg_only", "hybrid_prompt", "retrieval_prompt", "kg_prompt"}
    }

    rows = []
    for left_label, right_label in comparisons:
        left_records = loaded[left_label]
        right_records = loaded[right_label]
        for metric_name in METRICS:
            rows.append(
                summarize_metric_comparison(
                    left_records,
                    right_records,
                    metric_name=metric_name,
                    left_label=left_label,
                    right_label=right_label,
                    bootstrap_rounds=bootstrap_rounds,
                    seed=seed,
                )
            )
    return pd.DataFrame(rows)


def run_perturbation_comparisons(dataset_paths: dict[str, Path],
                                 bootstrap_rounds: int,
                                 seed: int) -> pd.DataFrame:
    datasets = [
        ("hybrid_perturb", "hybrid"),
        ("retrieval_perturb", "retrieval_only"),
        ("kg_perturb", "kg_only"),
    ]
    rows = []
    for dataset_key, label in datasets:
        records = load_jsonl(dataset_paths[dataset_key])
        for right_condition in ("E2", "E3", "E4"):
            left_records = filter_records(records, condition="E1")
            right_records = filter_records(records, condition=right_condition)
            for metric_name in METRICS:
                summary = summarize_metric_comparison(
                    left_records,
                    right_records,
                    metric_name=metric_name,
                    left_label=f"{label}:E1",
                    right_label=f"{label}:{right_condition}",
                    bootstrap_rounds=bootstrap_rounds,
                    seed=seed,
                )
                summary["dataset"] = label
                rows.append(summary)
    return pd.DataFrame(rows)


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        rendered = []
        for column in frame.columns:
            value = row[column]
            if pd.isna(value):
                rendered.append("")
            elif isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.6g}")
            else:
                rendered.append(str(value))
        rows.append(rendered)

    widths = [len(column) for column in columns]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render_row(values: list[str]) -> str:
        padded = [value.ljust(width) for value, width in zip(values, widths)]
        return "| " + " | ".join(padded) + " |"

    header = render_row(columns)
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [render_row(row) for row in rows]
    return "\n".join([header, divider, *body])


def write_markdown_report(main_df: pd.DataFrame,
                          perturb_df: pd.DataFrame,
                          dataset_paths: dict[str, Path],
                          output_path: Path) -> None:
    lines = [
        "# Phase 5 Significance Summary",
        "",
        "## Dataset Sources",
        "",
    ]
    for label, path in sorted(dataset_paths.items()):
        lines.append(f"- `{label}`: `{path}`")

    lines.extend(["", "## Main Comparisons", ""])
    lines.append(dataframe_to_markdown_table(main_df))
    lines.extend(["", "## Perturbation Comparisons", ""])
    lines.append(dataframe_to_markdown_table(perturb_df))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Phase 5 significance testing")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing MovieRecommendationRAG/result outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "phase5_stats",
        help="Directory for significance outputs",
    )
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_paths = resolve_default_datasets(args.results_root)
    main_df = run_main_comparisons(dataset_paths, args.bootstrap_rounds, args.seed)
    perturb_df = run_perturbation_comparisons(dataset_paths, args.bootstrap_rounds, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_csv = args.output_dir / "significance_main.csv"
    perturb_csv = args.output_dir / "significance_perturbation.csv"
    summary_json = args.output_dir / "significance_summary.json"
    summary_md = args.output_dir / "significance_summary.md"

    main_df.to_csv(main_csv, index=False)
    perturb_df.to_csv(perturb_csv, index=False)
    payload = {
        "datasets": {label: str(path) for label, path in dataset_paths.items()},
        "main_comparisons": main_df.to_dict(orient="records"),
        "perturbation_comparisons": perturb_df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(main_df, perturb_df, dataset_paths, summary_md)

    print(f"Wrote {main_csv}")
    print(f"Wrote {perturb_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
