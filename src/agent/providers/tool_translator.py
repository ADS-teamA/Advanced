"""
Tool schema translator: Anthropic ↔ OpenAI formats.

PRAGMATIC APPROACH:
- Primary direction: Anthropic → OpenAI (our tools are Anthropic-native)
- Simple mapping (no complex validation)
- Lossy conversion is acceptable (OpenAI has fewer features)

Schema Formats:

Anthropic (native):
{
    "name": "search_documents",
    "description": "Search documents using hybrid retrieval",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
}

OpenAI (target):
{
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search documents using hybrid retrieval",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
}
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ToolSchemaTranslator:
    """
    Translate tool schemas between Anthropic and OpenAI formats.

    Key Insight:
    Both providers use JSON Schema for parameters, just wrapped differently.
    Our Pydantic schemas (via model_json_schema()) work for both!
    """

    @staticmethod
    def to_openai(anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Anthropic tool format → OpenAI function format.

        Args:
            anthropic_tools: List of tools in Anthropic format

        Returns:
            List of tools in OpenAI function calling format

        Raises:
            ValueError: If tool structure is invalid

        Example:
            >>> translator = ToolSchemaTranslator()
            >>> openai_tools = translator.to_openai(anthropic_tools)
        """
        openai_tools = []

        for i, tool in enumerate(anthropic_tools):
            # Validate required fields
            if "name" not in tool:
                raise ValueError(f"Tool {i} missing 'name' field: {tool}")
            if "description" not in tool:
                raise ValueError(f"Tool {i} missing 'description' field: {tool}")
            if "input_schema" not in tool:
                raise ValueError(f"Tool {i} missing 'input_schema' field: {tool}")

            # Validate input_schema is a dict
            if not isinstance(tool["input_schema"], dict):
                raise ValueError(
                    f"Tool {i} input_schema must be dict, got {type(tool['input_schema'])}"
                )

            # Remove Anthropic-specific fields
            input_schema = tool["input_schema"].copy()

            # Remove cache_control if present (Anthropic-specific optimization)
            input_schema.pop("cache_control", None)

            # Convert to OpenAI format
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": input_schema,  # JSON Schema is compatible!
                },
            }

            openai_tools.append(openai_tool)

        logger.debug(f"Translated {len(anthropic_tools)} tools to OpenAI format")
        return openai_tools

    @staticmethod
    def to_anthropic(openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI function format → Anthropic tool format.

        Reverse direction (not needed for current use case but included for completeness).

        Args:
            openai_tools: List of tools in OpenAI function format

        Returns:
            List of tools in Anthropic format
        """
        anthropic_tools = []

        for tool in openai_tools:
            if tool.get("type") != "function":
                logger.warning(f"Skipping non-function tool: {tool.get('type')}")
                continue

            func = tool["function"]

            anthropic_tool = {
                "name": func["name"],
                "description": func["description"],
                "input_schema": func["parameters"],
            }

            anthropic_tools.append(anthropic_tool)

        logger.debug(f"Translated {len(openai_tools)} tools to Anthropic format")
        return anthropic_tools

    @staticmethod
    def remove_cache_control(obj: Any) -> Any:
        """
        Recursively remove cache_control from objects.

        OpenAI doesn't support prompt caching, so we strip these markers.

        Args:
            obj: Any object (dict, list, or primitive)

        Returns:
            Object with cache_control removed
        """
        if isinstance(obj, dict):
            return {k: ToolSchemaTranslator.remove_cache_control(v) for k, v in obj.items() if k != "cache_control"}
        elif isinstance(obj, list):
            return [ToolSchemaTranslator.remove_cache_control(item) for item in obj]
        else:
            return obj
