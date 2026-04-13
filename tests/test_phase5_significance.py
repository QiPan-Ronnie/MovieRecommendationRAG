import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from analysis.phase5_significance import (
    align_records_on_pairs,
    bootstrap_mean_difference,
    dataframe_to_markdown_table,
    summarize_metric_comparison,
    write_markdown_report,
)


class AlignRecordsOnPairsTests(unittest.TestCase):
    def test_align_records_uses_pair_intersection_and_normalizes_ids(self):
        left_records = [
            {"user_id": 1, "movie_id": 10, "bert_score_f1": 0.8},
            {"user_id": 1, "movie_id": 11, "bert_score_f1": 0.7},
        ]
        right_records = [
            {"user_id": 1.0, "movie_id": 10.0, "bert_score_f1": 0.6},
            {"user_id": 2.0, "movie_id": 99.0, "bert_score_f1": 0.1},
        ]

        aligned = align_records_on_pairs(left_records, right_records, "bert_score_f1")

        self.assertEqual(aligned["pair_keys"], [(1, 10)])
        self.assertEqual(aligned["left_values"], [0.8])
        self.assertEqual(aligned["right_values"], [0.6])


class BootstrapMeanDifferenceTests(unittest.TestCase):
    def test_bootstrap_ci_contains_mean_difference(self):
        diffs = [0.1, 0.2, 0.3, 0.4]
        result = bootstrap_mean_difference(diffs, num_rounds=500, seed=7)

        self.assertAlmostEqual(result["mean_difference"], 0.25)
        self.assertLessEqual(result["ci_low"], result["mean_difference"])
        self.assertGreaterEqual(result["ci_high"], result["mean_difference"])

    def test_summary_reports_pair_count_and_mean_difference(self):
        left_records = [
            {"user_id": 1, "movie_id": 10, "evidence_overlap": 0.5},
            {"user_id": 1, "movie_id": 11, "evidence_overlap": 0.4},
            {"user_id": 1, "movie_id": 12, "evidence_overlap": 0.3},
        ]
        right_records = [
            {"user_id": 1, "movie_id": 10, "evidence_overlap": 0.2},
            {"user_id": 1, "movie_id": 11, "evidence_overlap": 0.1},
            {"user_id": 1, "movie_id": 12, "evidence_overlap": 0.0},
        ]

        summary = summarize_metric_comparison(
            left_records,
            right_records,
            metric_name="evidence_overlap",
            left_label="hybrid",
            right_label="prompt_only",
            bootstrap_rounds=500,
            seed=7,
        )

        self.assertEqual(summary["pair_count"], 3)
        self.assertAlmostEqual(summary["left_mean"], 0.4)
        self.assertAlmostEqual(summary["right_mean"], 0.1)
        self.assertAlmostEqual(summary["mean_difference"], 0.3)


class MarkdownReportTests(unittest.TestCase):
    def test_markdown_report_does_not_require_tabulate(self):
        main_df = pd.DataFrame([
            {"left_label": "hybrid", "right_label": "kg_only", "metric": "bert_score_f1", "mean_difference": 0.01}
        ])
        perturb_df = pd.DataFrame([
            {"left_label": "hybrid:E1", "right_label": "hybrid:E4", "metric": "evidence_overlap", "mean_difference": 0.12}
        ])

        rendered = dataframe_to_markdown_table(main_df)
        self.assertIn("| left_label |", rendered)
        self.assertIn("hybrid", rendered)

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "summary.md"
            write_markdown_report(
                main_df,
                perturb_df,
                {"hybrid": Path("/tmp/hybrid.jsonl")},
                output_path,
            )
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("# Phase 5 Significance Summary", text)
        self.assertIn("## Main Comparisons", text)
        self.assertIn("| left_label |", text)


if __name__ == "__main__":
    unittest.main()
