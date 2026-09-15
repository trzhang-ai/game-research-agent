import json
import unittest
from types import SimpleNamespace

from pydantic import BaseModel

from lib.tooling import tool
from game_agent import GameResearchAgent


class EvaluationReport(BaseModel):
    useful: bool
    description: str


@tool(name="classify_request")
def route_request(route: str, retrieval_question: str = "") -> dict:
    """Return a deterministic route for testing."""
    return {"route": route, "retrieval_question": retrieval_question}


@tool(name="retrieve_game")
def retrieve_fixture(query: str) -> list[dict]:
    """Return a Unicode-bearing game fixture."""
    return [{"Name": "Pokémon Gold and Silver", "query": query}]


@tool(name="evaluate_retrieval")
def evaluate_fixture(useful: bool) -> EvaluationReport:
    """Return a deterministic sufficiency decision."""
    return EvaluationReport(useful=useful, description="fixture")


def make_tool_call(name: str, arguments: dict):
    return SimpleNamespace(
        id=f"call-{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def make_state(phase: str, call):
    return {
        "session_id": "test",
        "user_query": "question",
        "instructions": "test",
        "messages": [],
        "current_tool_calls": [call],
        "total_tokens": 0,
        "phase": phase,
        "evaluation_useful": None,
    }


class AgentRoutingTests(unittest.TestCase):
    def setUp(self):
        self.agent = GameResearchAgent(
            model_name="test-model",
            reasoning_effort="low",
            instructions="test",
            tools=[route_request, retrieve_fixture, evaluate_fixture],
        )

    def test_game_research_route_advances_to_local_retrieval(self):
        state = make_state(
            "route",
            make_tool_call(
                "classify_request",
                {
                    "route": "game_research",
                    "retrieval_question": "When was Pokémon released?",
                },
            ),
        )

        updated = self.agent._tool_step(state)

        self.assertEqual(updated["phase"], "retrieve")

    def test_insufficient_retrieval_advances_to_web(self):
        state = make_state(
            "evaluate",
            make_tool_call("evaluate_retrieval", {"useful": False}),
        )

        updated = self.agent._tool_step(state)

        self.assertEqual(updated["phase"], "web")
        self.assertFalse(updated["evaluation_useful"])

    def test_sufficient_retrieval_advances_to_answer(self):
        state = make_state(
            "evaluate",
            make_tool_call("evaluate_retrieval", {"useful": True}),
        )

        updated = self.agent._tool_step(state)

        self.assertEqual(updated["phase"], "answer")
        self.assertTrue(updated["evaluation_useful"])

    def test_tool_serialization_preserves_unicode(self):
        state = make_state(
            "retrieve",
            make_tool_call("retrieve_game", {"query": "Pokémon"}),
        )

        updated = self.agent._tool_step(state)
        tool_payload = updated["messages"][-1].content

        self.assertIn("Pokémon", tool_payload)
        self.assertNotIn("\\u00e9", tool_payload)
        self.assertEqual(updated["phase"], "evaluate")


if __name__ == "__main__":
    unittest.main()
