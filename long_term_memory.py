from lib.tooling import Tool
from lib.memory import LongTermMemory, MemoryFragment


def build_memory_registration_tool(ltm: LongTermMemory, owner: str, namespace: str):
    """
    Create a tool for agents to register new memories.

    This factory function creates a tool that allows AI agents to store new
    information about users in the long-term memory system. The tool is
    pre-configured with specific owner and namespace parameters.

    Args:
        ltm (LongTermMemory): The memory system instance to use
        owner (str): User identifier for memory ownership
        namespace (str): Namespace for organizing memories

    Returns:
        Tool: A configured tool for memory registration
    """

    def _register(content: str):
        ltm.register(MemoryFragment(content=content, owner=owner, namespace=namespace))
        return "Saved new memory"

    return Tool(
        func=_register,
        name="register_memory",
        description=(
            "Register a new memory or preference about the user, "
            "so it can be useful later as context.\n"
            "Args:\n"
            "    content: The information to save"
        ),
    )


def build_memory_search_tool(ltm: LongTermMemory, owner: str, namespace: str):
    """
    Create a tool for agents to search existing memories.

    This factory function creates a tool that allows AI agents to retrieve
    relevant memories from the long-term memory system based on semantic
    similarity to a search query.

    Args:
        ltm (LongTermMemory): The memory system instance to use
        owner (str): User identifier for memory ownership
        namespace (str): Namespace to search within

    Returns:
        Tool: A configured tool for memory search
    """

    def _search(query: str):
        result = ltm.search(
            query_text=query,
            owner=owner,
            namespace=namespace,
            limit=3,
        )

        distances = result.metadata.get("distances") or []

        return [
            {
                "content": fragment.content,
                "owner": fragment.owner,
                "namespace": fragment.namespace,
                "timestamp": fragment.timestamp,
                "distance": (distances[index] if index < len(distances) else None),
            }
            for index, fragment in enumerate(result.fragments)
        ]

    return Tool(
        func=_search,
        name="search_memory",
        description=(
            "Search stored long-term memory relevant to the user's request.\n"
            "Args:\n"
            "    query: What user-specific information to retrieve."
        ),
    )
