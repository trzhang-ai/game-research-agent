import unittest
from typing import Literal, TypedDict

from lib.tooling import tool


class Record(TypedDict):
    title: str
    year: int


class ToolSchemaTests(unittest.TestCase):
    def test_nested_types_and_routes_are_reflected_in_schema(self):
        @tool
        def inspect_records(
            route: Literal["local", "web"],
            records: list[Record],
            note: str | None = None,
        ) -> dict:
            """Inspect typed records."""
            return {"route": route, "records": records, "note": note}

        function_schema = inspect_records.model_dump()["function"]
        parameters = function_schema["parameters"]

        self.assertEqual(function_schema["name"], "inspect_records")
        self.assertEqual(
            parameters["properties"]["route"]["enum"], ["local", "web"]
        )
        record_schema = parameters["properties"]["records"]["items"]
        self.assertEqual(record_schema["type"], "object")
        self.assertEqual(record_schema["properties"]["year"]["type"], "integer")
        self.assertEqual(parameters["required"], ["route", "records"])

    def test_decorated_tool_remains_callable(self):
        @tool(name="add_numbers")
        def add(left: int, right: int) -> int:
            """Add two integers."""
            return left + right

        self.assertEqual(add(left=2, right=3), 5)
        self.assertEqual(add.name, "add_numbers")


if __name__ == "__main__":
    unittest.main()
