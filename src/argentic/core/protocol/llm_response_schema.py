"""JSON schemas for LLM responses following Argentic's structured format."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ToolCallSchema(BaseModel):
    """Schema for a single tool call within a tool_call response."""

    tool_id: str = Field(description="The exact tool_id from the available tools list")
    arguments: Dict[str, Any] = Field(
        description="Arguments dictionary matching the tool's required parameters"
    )


class DirectResponseSchema(BaseModel):
    """Schema for direct answer responses (no tools needed)."""

    type: Literal["direct"] = Field(description="Response type indicator")
    content: str = Field(description="The direct answer to the user's question")


class ToolCallResponseSchema(BaseModel):
    """Schema for tool call requests."""

    type: Literal["tool_call"] = Field(description="Response type indicator")
    tool_calls: List[ToolCallSchema] = Field(
        description="List of tools to execute with their arguments"
    )


class ToolResultResponseSchema(BaseModel):
    """Schema for final answers after tool execution."""

    type: Literal["tool_result"] = Field(description="Response type indicator")
    tool_id: str = Field(description="The tool_id of the executed tool")
    result: str = Field(
        description="Final answer incorporating tool results, or explanation if tool didn't help"
    )


class AgentResponseSchema(BaseModel):
    """Union schema accepting any of the three response types."""

    type: Literal["direct", "tool_call", "tool_result"] = Field(
        description="Response type: 'direct' for direct answers, 'tool_call' for tool execution requests, 'tool_result' for final answers after tool execution"
    )
    # Optional fields depending on type
    content: Optional[str] = Field(
        None, description="Answer content (for type='direct')"
    )
    tool_calls: Optional[List[ToolCallSchema]] = Field(
        None, description="Tool execution requests (for type='tool_call')"
    )
    tool_id: Optional[str] = Field(
        None, description="Executed tool ID (for type='tool_result')"
    )
    result: Optional[str] = Field(
        None, description="Final answer (for type='tool_result')"
    )


def get_agent_response_json_schema() -> Dict[str, Any]:
    """
    Get the JSON schema for agent responses.

    This schema enforces Argentic's structured response format for LLMs.
    Returns a simplified schema compatible with Gemini's JSON Schema subset (no $defs).

    Returns:
        JSON schema dictionary compatible with Gemini's response_json_schema parameter.
    """
    # Inline schema without $defs (Gemini doesn't support references)
    return {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["direct", "tool_call", "tool_result"],
                "description": "Response type: 'direct' for direct answers, 'tool_call' for tool execution requests, 'tool_result' for final answers after tool execution",
            },
            "content": {
                "type": "string",
                "description": "Answer content (required when type='direct')",
            },
            "tool_calls": {
                "type": "array",
                "description": "Tool execution requests (required when type='tool_call')",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool_id": {
                            "type": "string",
                            "description": "The exact tool_id from the available tools list",
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments dictionary matching the tool's required parameters",
                        },
                    },
                    "required": ["tool_id", "arguments"],
                },
            },
            "tool_id": {
                "type": "string",
                "description": "Executed tool ID (required when type='tool_result')",
            },
            "result": {
                "type": "string",
                "description": "Final answer (required when type='tool_result')",
            },
        },
        "required": ["type"],
    }


def get_direct_response_json_schema() -> Dict[str, Any]:
    """
    Get a simplified JSON schema for direct responses only.

    Use this when tools are not available or for simple Q&A scenarios.

    Returns:
        JSON schema dictionary for direct responses.
    """
    return DirectResponseSchema.model_json_schema()
