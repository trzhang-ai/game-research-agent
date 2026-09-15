import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "01_game_index.ipynb",
    ROOT / "02_research_agent.ipynb",
]


class NotebookQualityTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in ROOT.rglob("*.py"):
            if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
                continue
            with self.subTest(module=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_notebooks_are_valid_and_code_cells_parse(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)

                title = "".join(notebook["cells"][0]["source"])
                self.assertNotIn("STARTER", title.upper())

                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(cell.get("source", [])).strip()
                    self.assertTrue(source, f"Empty code cell {index} in {path.name}")
                    ast.parse(source, filename=f"{path.name}:cell-{index}")

    def test_notebooks_do_not_store_stale_execution_outputs(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    self.assertIsNone(
                        cell.get("execution_count"),
                        f"Execution count retained in cell {index} of {path.name}",
                    )
                    self.assertEqual(
                        cell.get("outputs", []),
                        [],
                        f"Output retained in cell {index} of {path.name}",
                    )


if __name__ == "__main__":
    unittest.main()
