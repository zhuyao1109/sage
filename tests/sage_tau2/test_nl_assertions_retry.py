"""NL assertions judge should tolerate empty / markdown-wrapped JSON via retries."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator


class NLAssertionsRetryTests(unittest.TestCase):
    def test_retries_empty_then_parses_markdown_json(self) -> None:
        responses = [
            SimpleNamespace(content=""),
            SimpleNamespace(
                content='```json\n{"results":[{"expectedOutcome":"ok","reasoning":"r","metExpectation":true}]}\n```'
            ),
        ]

        def fake_generate(**kwargs):
            return responses.pop(0)

        with patch(
            "tau2.evaluator.evaluator_nl_assertions.generate",
            side_effect=fake_generate,
        ):
            checks = NLAssertionsEvaluator.evaluate_nl_assertions(
                trajectory=[SimpleNamespace(role="user", content="hi")],
                nl_assertions=["ok"],
            )
        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0].met)
        self.assertEqual(checks[0].nl_assertion, "ok")

    def test_raises_after_exhausted_empty(self) -> None:
        with patch(
            "tau2.evaluator.evaluator_nl_assertions.generate",
            return_value=SimpleNamespace(content=""),
        ), patch(
            "tau2.evaluator.evaluator_nl_assertions._NL_JUDGE_RETRY_SLEEP_S",
            0,
        ):
            with self.assertRaises(RuntimeError):
                NLAssertionsEvaluator.evaluate_nl_assertions(
                    trajectory=[SimpleNamespace(role="user", content="hi")],
                    nl_assertions=["ok"],
                )


if __name__ == "__main__":
    unittest.main()
