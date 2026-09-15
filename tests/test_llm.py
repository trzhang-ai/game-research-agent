import unittest

from lib.llm import LLM
from lib.messages import AIMessage, TokenUsage, ToolMessage


class LLMPayloadTests(unittest.TestCase):
    def setUp(self):
        self.llm = LLM(model="test-model", api_key="test-key")

    def test_internal_message_metadata_is_not_sent_to_openai(self):
        messages = [
            AIMessage(
                content="Pokémon",
                token_usage=TokenUsage(total_tokens=12),
            ),
            ToolMessage(
                content='{"useful": true}',
                name="evaluate_retrieval",
                tool_call_id="call-1",
            ),
        ]

        payload = self.llm._build_payload(messages)

        self.assertEqual(payload["messages"][0], {
            "role": "assistant",
            "content": "Pokémon",
        })
        self.assertEqual(payload["messages"][1], {
            "role": "tool",
            "content": '{"useful": true}',
            "tool_call_id": "call-1",
        })


if __name__ == "__main__":
    unittest.main()
