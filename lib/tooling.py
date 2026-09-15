import datetime
import inspect
import types
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    TypeAlias,
    get_type_hints,
    get_origin,
    get_args,
    is_typeddict,
)

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)


# Type alias for OpenAI's tool call implementation
ToolCall: TypeAlias = ChatCompletionMessageToolCall


class Tool:
    """Callable wrapper that exposes an OpenAI function-tool schema."""

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func) or ""
        self.signature = inspect.signature(func, eval_str=True)
        self.type_hints = get_type_hints(func)

        self.parameters = [
            self._build_param_schema(key, param)
            for key, param in self.signature.parameters.items()
        ]

    def _build_param_schema(self, name: str, param: inspect.Parameter):
        param_type = self.type_hints.get(name, str)
        schema = self._infer_json_schema_type(param_type)
        return {
            "name": name,
            "schema": schema,
            "required": param.default == inspect.Parameter.empty,
        }

    def _infer_json_schema_type(self, typ: Any) -> dict:
        origin = get_origin(typ)

        if typ is Any:
            return {}

        if typ is dict:
            return {"type": "object", "additionalProperties": True}

        if origin is Literal:
            return {"type": "string", "enum": list(get_args(typ))}

        if origin in (Union, types.UnionType):
            args = get_args(typ)
            non_none = [arg for arg in args if arg is not type(None)]
            if len(non_none) == 1:
                return self._infer_json_schema_type(non_none[0])
            return {"anyOf": [self._infer_json_schema_type(arg) for arg in args]}

        if origin is list:
            return {
                "type": "array",
                "items": self._infer_json_schema_type(
                    get_args(typ)[0] if get_args(typ) else str
                ),
            }

        if origin is dict:
            return {
                "type": "object",
                "additionalProperties": self._infer_json_schema_type(
                    get_args(typ)[1] if get_args(typ) else str
                ),
            }

        if is_typeddict(typ):
            hints = get_type_hints(typ)
            required_keys = getattr(typ, "__required_keys__", set(hints))
            return {
                "type": "object",
                "properties": {
                    key: self._infer_json_schema_type(value)
                    for key, value in hints.items()
                },
                "required": [key for key in hints if key in required_keys],
                "additionalProperties": False,
            }

        mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            type(None): "null",
            datetime.date: "string",
            datetime.datetime: "string",
        }

        return {"type": mapping.get(typ, "string")}

    def model_dump(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param["name"]: param["schema"] for param in self.parameters
                    },
                    "required": [
                        param["name"] for param in self.parameters if param["required"]
                    ],
                    "additionalProperties": False,
                },
            },
        }

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return f"<Tool name={self.name} params={[p['name'] for p in self.parameters]}>"

    @classmethod
    def from_func(cls, func: Callable) -> "Tool":
        return cls(func)


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    """Decorate a callable as a :class:`Tool`."""

    def wrapper(f):
        return Tool(f, name=name, description=description)

    # Support both ``@tool`` and ``@tool(name="custom_name")``.
    return wrapper(func) if func else wrapper
