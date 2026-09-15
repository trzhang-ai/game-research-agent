import unittest

from lib.evaluation import AgentEvaluator
from lib.messages import UserMessage


class AgentEvaluatorTests(unittest.TestCase):
    def test_missing_expected_tool_call_returns_explicit_failure(self):
        evaluator = AgentEvaluator.__new__(AgentEvaluator)

        result = evaluator.evaluate_single_step(
            agent_messages=[UserMessage(content="Question")],
            expected_tool_calls=["retrieve_game"],
        )

        self.assertEqual(result.overall_score, 0.0)
        self.assertFalse(result.task_completion.task_completed)
        self.assertFalse(result.quality_control.format_correct)


if __name__ == "__main__":
    unittest.main()
