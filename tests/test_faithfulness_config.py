import unittest

from rag.faithfulness import resolve_bertscore_scorer_kwargs


class ResolveBERTScoreScorerKwargsTests(unittest.TestCase):
    def test_local_model_defaults_to_no_rescale(self):
        model_label, kwargs = resolve_bertscore_scorer_kwargs(
            use_local_model=True,
            local_roberta_dir="/models/roberta-large",
            rescale_with_baseline=None,
        )

        self.assertEqual(model_label, "/models/roberta-large")
        self.assertEqual(kwargs["model_type"], "/models/roberta-large")
        self.assertEqual(kwargs["num_layers"], 17)
        self.assertIsNone(kwargs["lang"])
        self.assertFalse(kwargs["rescale_with_baseline"])

    def test_remote_model_defaults_to_rescale(self):
        model_label, kwargs = resolve_bertscore_scorer_kwargs(
            use_local_model=False,
            local_roberta_dir="/models/roberta-large",
            rescale_with_baseline=None,
        )

        self.assertEqual(model_label, "roberta-large")
        self.assertEqual(kwargs["lang"], "en")
        self.assertTrue(kwargs["rescale_with_baseline"])

    def test_explicit_override_wins(self):
        model_label, kwargs = resolve_bertscore_scorer_kwargs(
            use_local_model=False,
            local_roberta_dir="/models/roberta-large",
            rescale_with_baseline=False,
        )

        self.assertEqual(model_label, "roberta-large")
        self.assertFalse(kwargs["rescale_with_baseline"])


if __name__ == "__main__":
    unittest.main()
