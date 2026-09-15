import unittest

from lib.memory import LongTermMemory, MemoryFragment, ShortTermMemory


class FakeVectorStore:
    def __init__(self):
        self.upserted = []
        self.last_query = None

    def upsert(self, document):
        self.upserted.append(document)

    def query(self, **kwargs):
        self.last_query = kwargs
        return {
            "documents": [["Stored preference"]],
            "metadatas": [[
                {
                    "owner": "owner-1",
                    "namespace": "games",
                    "timestamp": 123,
                }
            ]],
            "distances": [[0.12]],
        }


class FakeVectorStoreManager:
    def __init__(self):
        self.store = FakeVectorStore()

    def get_or_create_store(self, name):
        if name != "long_term_memory":
            raise AssertionError(f"Unexpected store name: {name}")
        return self.store


class ShortTermMemoryTests(unittest.TestCase):
    def test_sessions_are_isolated_and_values_are_copied(self):
        memory = ShortTermMemory()
        memory.create_session("alpha")
        memory.create_session("beta")

        original = {"messages": ["first"]}
        memory.add(original, "alpha")
        original["messages"].append("mutated")

        retrieved = memory.get_last_object("alpha")
        self.assertEqual(retrieved, {"messages": ["first"]})
        self.assertIsNone(memory.get_last_object("beta"))

        retrieved["messages"].append("changed copy")
        self.assertEqual(
            memory.get_last_object("alpha"), {"messages": ["first"]}
        )


class LongTermMemoryTests(unittest.TestCase):
    def setUp(self):
        self.manager = FakeVectorStoreManager()
        self.memory = LongTermMemory(self.manager)

    def test_register_uses_a_stable_id_and_upsert(self):
        first = MemoryFragment(
            content="Pokémon is my favorite series.",
            owner="owner-1",
            namespace="games",
            timestamp=100,
        )
        second = MemoryFragment(
            content="Pokémon is my favorite series.",
            owner="owner-1",
            namespace="games",
            timestamp=200,
        )

        first_id = self.memory.register(first)
        second_id = self.memory.register(second)

        self.assertEqual(first_id, second_id)
        self.assertEqual(len(self.manager.store.upserted), 2)
        self.assertEqual(self.manager.store.upserted[0].id, first_id)
        self.assertEqual(
            self.manager.store.upserted[1].metadata["timestamp"], 200
        )

    def test_search_applies_owner_and_namespace_filters(self):
        result = self.memory.search(
            query_text="What do I like?",
            owner="owner-1",
            namespace="games",
            limit=1,
        )

        self.assertEqual(result.fragments[0].content, "Stored preference")
        self.assertEqual(result.metadata["distances"], [0.12])
        self.assertEqual(
            self.manager.store.last_query["where"],
            {
                "$and": [
                    {"namespace": {"$eq": "games"}},
                    {"owner": {"$eq": "owner-1"}},
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
