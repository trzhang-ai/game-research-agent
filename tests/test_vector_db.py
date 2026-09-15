import tempfile
import unittest
from unittest.mock import patch

from lib.documents import Corpus, Document
from lib.vector_db import VectorStore, VectorStoreManager


class FakeCollection:
    def __init__(self):
        self.last_upsert = None

    def upsert(self, **kwargs):
        self.last_upsert = kwargs


class VectorStoreTests(unittest.TestCase):
    def test_upsert_normalizes_a_corpus_for_chroma(self):
        collection = FakeCollection()
        store = VectorStore(collection)

        store.upsert(
            Corpus(
                [
                    Document(
                        id="game-1",
                        content="Pokémon",
                        metadata={"platform": "Game Boy Color"},
                    )
                ]
            )
        )

        self.assertEqual(collection.last_upsert["ids"], ["game-1"])
        self.assertEqual(collection.last_upsert["documents"], ["Pokémon"])
        self.assertEqual(
            collection.last_upsert["metadatas"],
            [{"platform": "Game Boy Color"}],
        )


class VectorStoreManagerTests(unittest.TestCase):
    def test_collection_lifecycle_has_explicit_outcomes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"OPENAI_API_KEY": "test-key"}
        ):
            manager = VectorStoreManager("test-key", persist_path=directory)

            manager.create_store("test_store")
            self.assertIsNotNone(manager.get_store("test_store"))

            with self.assertRaisesRegex(ValueError, "already exists"):
                manager.create_store("test_store")

            self.assertTrue(manager.delete_store("test_store"))
            self.assertFalse(manager.delete_store("test_store"))


if __name__ == "__main__":
    unittest.main()
