import json
from lib.llm import LLM
from lib.tooling import Tool, ToolCall
from lib.memory import ShortTermMemory
from lib.messages import AIMessage, UserMessage, SystemMessage, ToolMessage
from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run
from pydantic import BaseModel
from typing import TypedDict, Optional, Union, Literal


# Define the state schema
class AgentState(TypedDict):
    session_id: str
    user_query: str  # The current user query being processed
    instructions: str  # System instructions for the agent
    messages: list[dict]  # list of conversation messages
    current_tool_calls: Optional[list[ToolCall]]  # Current pending tool calls
    total_tokens: int  # Track the cumulative total
    phase: Literal["route", "memory", "retrieve", "evaluate", "web", "answer"]
    evaluation_useful: Optional[bool]


class UdaAgent:
    def __init__(
        self,
        model_name: str,
        reasoning_effort: str,
        instructions: str,
        tools: list[Tool] = None,
    ):
        """
        Initialize an Agent

        Args:
            model_name: Name/identifier of the LLM model to use
            instructions: System instructions for the agent
            tools: Optional list of tools available to the agent
        """
        self.instructions = instructions
        self.tools = tools if tools else []
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort

        # Initialize memory and state machine
        self.short_term_memory = ShortTermMemory()
        self.workflow = self._create_state_machine()

    def _prepare_messages_step(self, state: AgentState) -> AgentState:
        """Step logic: Prepare messages for LLM consumption"""
        messages = state.get("messages", [])

        # If no messages exist, start with system message
        if not messages:
            messages = [SystemMessage(content=state["instructions"])]

        # Add the new user message
        messages.append(UserMessage(content=state["user_query"]))

        return {"messages": messages, "session_id": state["session_id"]}

    def _llm_step(self, state: AgentState) -> AgentState:
        """Step logic: Process the current state through the LLM"""
        phase = state["phase"]

        tool_name_by_phase = {
            "route": "classify_request",
            "memory": "search_memory",
            "retrieve": "retrieve_game",
            "evaluate": "evaluate_retrieval",
            "web": "game_web_search",
        }

        if phase == "answer":
            active_tools = []
            tool_choice = "none"
        else:
            tool_name = tool_name_by_phase[phase]
            active_tool = next(
                (tool for tool in self.tools if tool.name == tool_name), None
            )
            if active_tool is None:
                raise RuntimeError(f"Required tool is missing: {tool_name}")

            active_tools = [active_tool]
            tool_choice = {"type": "function", "function": {"name": tool_name}}

        # Initialize LLM
        llm = LLM(
            model=self.model_name,
            tools=active_tools,
            reasoning_effort=self.reasoning_effort,
        )

        response = llm.invoke(state["messages"], tool_choice=tool_choice)
        tool_calls = response.tool_calls if response.tool_calls else None

        current_total = state.get("total_tokens", 0)
        if response.token_usage:
            current_total += response.token_usage.total_tokens

        # Create AI message with content and tool calls
        ai_message = AIMessage(
            content=response.content,
            tool_calls=tool_calls,
        )

        return {
            "messages": state["messages"] + [ai_message],
            "current_tool_calls": tool_calls,
            "session_id": state["session_id"],
            "total_tokens": current_total,
        }

    def _tool_step(self, state: AgentState) -> AgentState:
        """Step logic: Execute any pending tool calls"""
        tool_calls = state["current_tool_calls"] or []
        tool_messages = []
        next_phase = state["phase"]
        evaluation_useful = state.get("evaluation_useful")

        for call in tool_calls:
            # Access tool call data correctly
            function_name = call.function.name
            function_args = json.loads(call.function.arguments)
            tool_call_id = call.id
            # Find the matching tool
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

            if function_name == "classify_request":
                selected_route = serializable_result.get("route")
                if selected_route == "game_research":
                    next_phase = "retrieve"
                elif selected_route == "long_term_memory":
                    next_phase = "memory"
                else:
                    next_phase = "answer"
            elif function_name == "search_memory":
                next_phase = "answer"
            elif function_name == "retrieve_game":
                next_phase = "evaluate"
            elif function_name == "evaluate_retrieval":
                useful = serializable_result.get("useful")
                if not isinstance(useful, bool):
                    raise RuntimeError(
                        "evaluate_retrieval did not return a valid useful value."
                    )
                evaluation_useful = useful
                next_phase = "answer" if useful else "web"
            elif function_name == "game_web_search":
                next_phase = "answer"

            tool_content = (
                serializable_result
                if isinstance(serializable_result, str)
                else json.dumps(serializable_result, ensure_ascii=False)
            )

            tool_messages.append(
                ToolMessage(
                    content=tool_content, tool_call_id=tool_call_id, name=function_name
                )
            )

        # Clear tool calls and add results to messages
        return {
            "messages": state["messages"] + tool_messages,
            "current_tool_calls": None,
            "session_id": state["session_id"],
            "phase": next_phase,
            "evaluation_useful": evaluation_useful,
        }

    def _create_state_machine(self) -> StateMachine[AgentState]:
        """Create the internal state machine for the agent"""
        machine = StateMachine[AgentState](AgentState)

        # Create steps
        entry = EntryPoint[AgentState]()
        message_prep = Step[AgentState]("message_prep", self._prepare_messages_step)
        llm_processor = Step[AgentState]("llm_processor", self._llm_step)
        tool_executor = Step[AgentState]("tool_executor", self._tool_step)
        termination = Termination[AgentState]()

        machine.add_steps(
            [entry, message_prep, llm_processor, tool_executor, termination]
        )

        # Add transitions
        machine.connect(entry, message_prep)
        machine.connect(message_prep, llm_processor)

        # Transition based on whether there are tool calls
        def check_tool_calls(state: AgentState) -> Union[Step[AgentState], str]:
            """Transition logic: Check if there are tool calls"""
            if state.get("current_tool_calls"):
                return tool_executor
            return termination

        machine.connect(llm_processor, [tool_executor, termination], check_tool_calls)
        machine.connect(
            tool_executor, llm_processor
        )  # Go back to llm after tool execution

        return machine

    def invoke(self, query: str, session_id: Optional[str] = None) -> Run:
        """
        Run the agent on a query

        Args:
            query: The user's query to process
            session_id: Optional session identifier (uses 'default' if None)

        Returns:
            The final run object after processing
        """
        session_id = session_id or "default"

        # Create session if it doesn't exist
        self.short_term_memory.create_session(session_id)

        # Get previous messages from last run if available
        previous_messages = []
        last_run: Run = self.short_term_memory.get_last_object(session_id)
        if last_run:
            last_state = last_run.get_final_state()
            if last_state:
                previous_messages = last_state["messages"]

        initial_state: AgentState = {
            "session_id": session_id,
            "user_query": query,
            "instructions": self.instructions,
            "messages": previous_messages,
            "current_tool_calls": None,
            "total_tokens": 0,
            "phase": "route",
            "evaluation_useful": None,
        }

        run_object = self.workflow.run(initial_state)

        # Store the complete run object in memory
        self.short_term_memory.add(run_object, session_id)

        return run_object

    def get_session_runs(self, session_id: Optional[str] = None) -> list[Run]:
        """Get all Run objects for a session

        Args:
            session_id: Optional session ID (uses 'default' if None)

        Returns:
            list of Run objects in the session
        """
        return self.short_term_memory.get_all_objects(session_id)

    def reset_session(self, session_id: Optional[str] = None):
        """Reset memory for a specific session

        Args:
            session_id: Optional session to reset (uses 'default' if None)
        """
        self.short_term_memory.reset(session_id)
