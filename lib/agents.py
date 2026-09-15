import json
from typing import List, Optional, TypedDict, Union

from pydantic import BaseModel

from lib.llm import LLM
from lib.memory import ShortTermMemory
from lib.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage, UserMessage
from lib.state_machine import EntryPoint, Run, StateMachine, Step, Termination
from lib.tooling import Tool, ToolCall


class AgentState(TypedDict):
    session_id: str
    user_query: str
    instructions: str
    messages: List[AnyMessage]
    current_tool_calls: Optional[List[ToolCall]]
    total_tokens: int


class Agent:
    """Generic tool-calling agent with session-scoped conversation history."""

    def __init__(
        self,
        model_name: str,
        instructions: str,
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """Initialize the agent.

        Args:
            model_name: OpenAI model identifier.
            instructions: System instructions applied to each new session.
            tools: Optional callable tools.
            temperature: Optional sampling temperature for compatible models.
            reasoning_effort: Optional reasoning effort for compatible models.
        """
        self.instructions = instructions
        self.tools = list(tools or [])
        self.model_name = model_name
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

        self.memory = ShortTermMemory()
        self.workflow = self._create_state_machine()

    def _prepare_messages_step(self, state: AgentState) -> AgentState:
        """Append a user request without mutating stored history."""
        messages = list(state.get("messages", []))

        if not messages:
            messages = [SystemMessage(content=state["instructions"])]

        messages.append(UserMessage(content=state["user_query"]))

        return {"messages": messages, "session_id": state["session_id"]}

    def _llm_step(self, state: AgentState) -> AgentState:
        """Run one model step with the configured tool set."""
        llm = LLM(
            model=self.model_name,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            tools=self.tools,
        )

        response = llm.invoke(state["messages"])
        tool_calls = response.tool_calls if response.tool_calls else None

        current_total = state.get("total_tokens", 0)
        if response.token_usage:
            current_total += response.token_usage.total_tokens

        ai_message = AIMessage(
            content=response.content,
            tool_calls=tool_calls,
            token_usage=response.token_usage,
        )

        return {
            "messages": state["messages"] + [ai_message],
            "current_tool_calls": tool_calls,
            "session_id": state["session_id"],
            "total_tokens": current_total,
        }

    def _tool_step(self, state: AgentState) -> AgentState:
        """Validate, execute, and serialize pending tool calls."""
        tool_calls = state["current_tool_calls"] or []
        tool_messages = []

        for call in tool_calls:
            function_name = call.function.name
            function_args = json.loads(call.function.arguments)
            tool_call_id = call.id
            tool = next((t for t in self.tools if t.name == function_name), None)

            if tool is None:
                raise RuntimeError(f"Unknown tool: {function_name}")

            try:
                tool.signature.bind(**function_args)
            except TypeError as exc:
                raise RuntimeError(
                    f"Invalid arguments from {function_name}: {function_args}"
                ) from exc

            raw_result = tool(**function_args)
            serializable_result = (
                raw_result.model_dump()
                if isinstance(raw_result, BaseModel)
                else raw_result
            )
            content = (
                serializable_result
                if isinstance(serializable_result, str)
                else json.dumps(serializable_result, ensure_ascii=False)
            )
            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name=function_name,
                )
            )

        return {
            "messages": state["messages"] + tool_messages,
            "current_tool_calls": None,
            "session_id": state["session_id"],
        }

    def _create_state_machine(self) -> StateMachine[AgentState]:
        """Build the model/tool execution loop."""
        machine = StateMachine[AgentState](AgentState)

        entry = EntryPoint[AgentState]()
        message_prep = Step[AgentState]("message_prep", self._prepare_messages_step)
        llm_processor = Step[AgentState]("llm_processor", self._llm_step)
        tool_executor = Step[AgentState]("tool_executor", self._tool_step)
        termination = Termination[AgentState]()
        
        machine.add_steps(
            [entry, message_prep, llm_processor, tool_executor, termination]
        )

        machine.connect(entry, message_prep)
        machine.connect(message_prep, llm_processor)

        def check_tool_calls(state: AgentState) -> Union[Step[AgentState], str]:
            """Continue while the model requests a tool."""
            if state.get("current_tool_calls"):
                return tool_executor
            return termination

        machine.connect(llm_processor, [tool_executor, termination], check_tool_calls)
        machine.connect(tool_executor, llm_processor)

        return machine

    def invoke(self, query: str, session_id: Optional[str] = None) -> Run:
        """Run one request while preserving history within its session.

        Args:
            query: User request to process.
            session_id: Session identifier; defaults to ``default``.

        Returns:
            Completed state-machine run.
        """
        session_id = session_id or "default"

        self.memory.create_session(session_id)

        previous_messages = []
        last_run: Run = self.memory.get_last_object(session_id)
        if last_run:
            last_state = last_run.get_final_state()
            if last_state:
                previous_messages = last_state["messages"]

        initial_state: AgentState = {
            "user_query": query,
            "instructions": self.instructions,
            "messages": previous_messages,
            "current_tool_calls": None,
            "session_id": session_id,
            "total_tokens": 0,
        }

        run_object = self.workflow.run(initial_state)

        self.memory.add(run_object, session_id)

        return run_object

    def get_session_runs(self, session_id: Optional[str] = None) -> List[Run]:
        """Return all runs for a session.

        Args:
            session_id: Session identifier; defaults to ``default``.

        Returns:
            Defensive copies of recorded runs.
        """
        return self.memory.get_all_objects(session_id)

    def reset_session(self, session_id: Optional[str] = None) -> None:
        """Clear one session, or all sessions when no ID is supplied.

        Args:
            session_id: Session identifier. ``None`` resets all sessions.
        """
        self.memory.reset(session_id)
