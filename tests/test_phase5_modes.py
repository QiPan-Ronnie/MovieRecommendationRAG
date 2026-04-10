import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from rag.generator import ExplanationGenerator, ExplanationRequest, LLMBackend
from rag.pipeline import (
    _build_evidence_bundle,
    _filter_recommendations_for_kg_paths,
    _get_primary_mode_label,
    _perturb_evidence,
    _pick_reference_explanation_path,
    run_phase_5_4,
)


class DummyLLM(LLMBackend):
    def __init__(self):
        self.calls = []

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
        self.calls.append((system_prompt, user_prompt, max_new_tokens))
        return "stub explanation"


class Phase5ModeTests(unittest.TestCase):
    def test_get_primary_mode_label_maps_evidence_modes(self):
        self.assertEqual(_get_primary_mode_label("hybrid"), "rag")
        self.assertEqual(_get_primary_mode_label("kg_only"), "kg_only")
        self.assertEqual(_get_primary_mode_label("retrieval_only"), "retrieval_only")

    def test_build_evidence_bundle_prepends_kg_for_hybrid(self):
        retrieved = [{"text": "retrieval", "source": "overview"}]
        kg = [{"text": "kg", "source": "kg_path"}]
        combined = _build_evidence_bundle(retrieved, kg, evidence_mode="hybrid")
        self.assertEqual(combined, kg + retrieved)

    def test_build_evidence_bundle_returns_only_kg_for_kg_only(self):
        retrieved = [{"text": "retrieval", "source": "overview"}]
        kg = [{"text": "kg", "source": "kg_path"}]
        combined = _build_evidence_bundle(retrieved, kg, evidence_mode="kg_only")
        self.assertEqual(combined, kg)

    def test_build_evidence_bundle_returns_only_retrieval_for_retrieval_only(self):
        retrieved = [{"text": "retrieval", "source": "overview"}]
        kg = [{"text": "kg", "source": "kg_path"}]
        combined = _build_evidence_bundle(retrieved, kg, evidence_mode="retrieval_only")
        self.assertEqual(combined, retrieved)

    def test_pick_reference_explanation_path_prefers_evidence_backed_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rag = tmp / "explanations_rag.jsonl"
            kg = tmp / "explanations_kg_only.jsonl"
            retrieval = tmp / "explanations_retrieval_only.jsonl"

            kg.write_text("{}\n", encoding="utf-8")
            retrieval.write_text("{}\n", encoding="utf-8")
            self.assertEqual(_pick_reference_explanation_path(str(tmp)), str(kg))

            rag.write_text("{}\n", encoding="utf-8")
            self.assertEqual(_pick_reference_explanation_path(str(tmp)), str(rag))

    def test_filter_recommendations_for_kg_paths_keeps_only_pairs_with_paths(self):
        recs = pd.DataFrame([
            {"user_id": 1, "movie_id": 10},
            {"user_id": 2, "movie_id": 20},
        ])
        filtered, skipped = _filter_recommendations_for_kg_paths(
            recs, {"1_10": [{}]}, evidence_mode="kg_only"
        )
        self.assertEqual(skipped, 1)
        self.assertEqual(filtered[["user_id", "movie_id"]].to_dict("records"), [{"user_id": 1, "movie_id": 10}])

    def test_kg_only_e4_uses_unrelated_kg_evidence(self):
        movie_info = {
            10: {"title": "Target Movie"},
            20: {"title": "Bridge Movie"},
            30: {"title": "Other Movie"},
            40: {"title": "History Movie"},
        }
        kg_paths = {
            "1_10": [
                {
                    "history_movie": 40,
                    "path": [
                        {"to": "genre_Drama", "relation": "has_genre"},
                        {"to": "movie_10", "relation": "has_genre"},
                    ],
                }
            ],
            "2_30": [
                {
                    "history_movie": 40,
                    "path": [
                        {"to": "movie_20", "relation": "co_liked"},
                        {"to": "movie_30", "relation": "co_liked"},
                    ],
                }
            ],
        }
        original = [{"text": "target evidence", "source": "kg_path", "movie_id": 10}]
        perturbed = _perturb_evidence(
            "E4",
            original,
            "kg_only",
            uid=1,
            mid=10,
            kg_paths=kg_paths,
            movie_info=movie_info,
        )
        self.assertTrue(perturbed)
        self.assertTrue(all(item["source"] == "kg_path" for item in perturbed))
        self.assertTrue(all(item["movie_id"] != 10 for item in perturbed))

    def test_run_phase_5_4_evaluates_kg_only_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            kg_path = tmp / "explanations_kg_only.jsonl"
            prompt_path = tmp / "explanations_prompt_only.jsonl"
            kg_record = {
                "user_id": 1,
                "movie_id": 10,
                "explanation": "kg explanation",
                "reference_evidence": [{"text": "kg evidence", "source": "kg_path"}],
            }
            prompt_record = {
                "user_id": 1,
                "movie_id": 10,
                "explanation": "prompt explanation",
                "evidence_used": [],
            }
            kg_path.write_text(json.dumps(kg_record) + "\n", encoding="utf-8")
            prompt_path.write_text(json.dumps(prompt_record) + "\n", encoding="utf-8")

            saved_dirs = []

            def fake_evaluate(records):
                return [{"condition": r["condition"], "reference_evidence": r.get("reference_evidence", [])} for r in records]

            def fake_aggregate(results):
                return {"count": len(results)}

            def fake_save(results, summary, output_dir):
                saved_dirs.append(output_dir)

            with patch("rag.faithfulness.evaluate_faithfulness", side_effect=fake_evaluate), \
                 patch("rag.faithfulness.aggregate_results", side_effect=fake_aggregate), \
                 patch("rag.faithfulness.save_faithfulness_results", side_effect=fake_save):
                run_phase_5_4(output_dir=str(tmp))

            self.assertIn(str(tmp / "faithfulness_kg_only"), saved_dirs)
            self.assertIn(str(tmp / "faithfulness_prompt_only"), saved_dirs)

    def test_kg_only_generation_uses_evidence_grounded_prompt(self):
        llm = DummyLLM()
        generator = ExplanationGenerator(llm)
        request = ExplanationRequest(
            user_id=1,
            candidate_movie_id=10,
            candidate_title="Example Movie",
            candidate_genres="Drama",
            history_titles=["History A"],
            evidence=[{"text": "KG evidence text", "source": "kg_path"}],
        )

        result = generator.generate_explanation(request, mode="kg_only")

        self.assertEqual(result.mode, "kg_only")
        self.assertEqual(result.evidence_used, request.evidence)
        self.assertTrue(llm.calls)
        system_prompt, user_prompt, _ = llm.calls[0]
        self.assertIn("grounded", system_prompt.lower())
        self.assertIn("Retrieved Evidence", user_prompt)
        self.assertIn("KG evidence text", user_prompt)


if __name__ == "__main__":
    unittest.main()
