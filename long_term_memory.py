from lib.tooling import Tool
from lib.memory import LongTermMemory


def build_memory_search_tool(
    ltm: LongTermMemory, owner: str, namespace: str
) -> Tool:
    """Build a read-only memory-search tool scoped to one owner and namespace.

    Args:
        ltm: Long-term-memory service to query.
        owner: User identifier applied as a mandatory filter.
        namespace: Logical memory partition applied as a mandatory filter.

    Returns:
        A callable tool that returns matching fragments and distances.
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
