from typing import Optional, Dict, Any
from pydantic import BaseModel
from openai import OpenAI
from lib.messages import (
    TokenUsage,
    AIMessage,
    BaseMessage,
    UserMessage,
)
from lib.tooling import Tool


class LLM:
    def __init__(
        self,
        model: str,
        reasoning_effort: str = None,
        tools: Optional[list[Tool]] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.tools: Dict[str, Tool] = {tool.name: tool for tool in (tools or [])}

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def _build_payload(
        self, messages: list[BaseMessage], tool_choice: str | Dict[str, Any] = "auto"
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
        }

        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        if self.tools:
            payload["tools"] = [tool.model_dump() for tool in self.tools.values()]
            payload["tool_choice"] = tool_choice
            payload["parallel_tool_calls"] = False

        return payload

    def _convert_input(self, input: Any) -> list[BaseMessage]:
        if isinstance(input, str):
            return [UserMessage(content=input)]
        elif isinstance(input, BaseMessage):
            return [input]
        elif isinstance(input, list) and all(isinstance(m, BaseMessage) for m in input):
            return input
        else:
            raise ValueError(f"Invalid input type {type(input)}.")

    def invoke(
        self,
        input: str | BaseMessage | list[BaseMessage],
        response_format: BaseModel = None,
        tool_choice: str | Dict[str, Any] = "auto",
    ) -> AIMessage:
        messages = self._convert_input(input)
        payload = self._build_payload(messages, tool_choice)
        if response_format:
            payload.update({"response_format": response_format})
            response = self.client.beta.chat.completions.parse(**payload)
        else:
            response = self.client.chat.completions.create(**payload)
        choice = response.choices[0]
        message = choice.message

        token_usage = None
        if response.usage:
            token_usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return AIMessage(
            content=message.content,
            tool_calls=message.tool_calls,
            token_usage=token_usage,
        )
