"""Shared test fixtures for function-call detector unit tests.

Provides ToolParserTestConfig with common Tool factory methods
used across all detector test files.
"""

from sgl_jax.srt.entrypoints.openai.protocol import Function, Tool


class ToolParserTestConfig:
    """Common factory methods for building Tool objects in detector tests."""

    @staticmethod
    def make_tool(name: str, properties: dict | None = None) -> Tool:
        return Tool(
            type="function",
            function=Function(
                name=name,
                description=f"{name} tool",
                parameters={"type": "object", "properties": properties or {}},
            ),
        )

    @staticmethod
    def bash_tool() -> Tool:
        return ToolParserTestConfig.make_tool(
            "execute_bash",
            {"command": {"type": "string"}},
        )

    @staticmethod
    def weather_tool() -> Tool:
        return ToolParserTestConfig.make_tool(
            "get_weather",
            {"location": {"type": "string"}},
        )

    @staticmethod
    def square_tool() -> Tool:
        return ToolParserTestConfig.make_tool(
            "square",
            {"x": {"type": "number"}},
        )

    @staticmethod
    def typed_tool() -> Tool:
        return ToolParserTestConfig.make_tool(
            "do_thing",
            {
                "n": {"type": "integer"},
                "ratio": {"type": "number"},
                "flag": {"type": "boolean"},
                "obj": {"type": "object"},
            },
        )
