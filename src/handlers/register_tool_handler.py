import uuid
import json
from typing import Dict, Any, Optional
from pydantic import (
    BaseModel,
)

from core.messager import Messager, MQTTMessage
from core.agent import Agent
from core.protocol.message import ToolRegisteredMessage


def handle_register_tool(
    messager: Messager,
    agent: Agent,
    data: Dict[str, Any],
    msg: MQTTMessage,
    handler_kwargs: Dict[str, Any],
) -> None:
    try:
        tool_name = data.get("tool_name")
        tool_manual = data.get("tool_manual")
        tool_api_schema_str = data.get("tool_api")
        source = data.get("source")

        if not all([tool_name, tool_manual, tool_api_schema_str, source]):
            raise ValueError(
                "Missing required fields (tool_name, tool_manual, tool_api, source) in registration message data."
            )

        argument_schema: Optional[type[BaseModel]] = None
        try:
            schema_dict = json.loads(tool_api_schema_str)
            if (
                isinstance(schema_dict, dict)
                and schema_dict.get("type") == "object"
                and "properties" in schema_dict
            ):
                messager.log(
                    f"Received schema for {tool_name}, dynamic model creation skipped for now."
                )
                pass
            else:
                messager.log(
                    f"Warning: tool_api for {tool_name} is not a valid JSON Schema object.",
                    level="warning",
                )

        except json.JSONDecodeError:
            messager.log(f"Warning: tool_api for {tool_name} is not valid JSON.", level="warning")
        except Exception as e:
            messager.log(f"Error processing schema for {tool_name}: {e}", level="error")

        def placeholder_implementation(**kwargs):
            messager.log(
                f"Placeholder tool '{tool_name}' called with args: {kwargs}. Not implemented.",
                level="warning",
            )
            return f"Error: Tool '{tool_name}' is registered but not fully implemented."

        tool_id = str(uuid.uuid4())

        agent.tool_manager.register_tool(
            tool_id=tool_id,
            tool_name=tool_name,
            tool_manual=tool_manual,
            argument_schema=argument_schema,
            implementation=placeholder_implementation,
        )

        print(f"Registered placeholder tool '{tool_name}' (ID: {tool_id}) from source '{source}'.")

        confirmation_message = ToolRegisteredMessage(
            source=handler_kwargs.get("mqtt_client_id", "rag_agent"),
            tool_id=tool_id,
            tool_name=tool_name,
            recipient=source,
        )
        response_topic = handler_kwargs.get("pub_status_topic", "rag/status/info")
        messager.publish(response_topic, confirmation_message.to_dict())
        messager.log(
            f"Sent TOOL_REGISTERED confirmation for tool '{tool_name}' to topic '{response_topic}' for recipient '{source}'."
        )

    except ValueError as ve:
        error_msg = f"Error processing REGISTER_TOOL message: {ve}"
        print(error_msg)
        messager.log(error_msg, level="error")
    except Exception as e:
        error_msg = f"Unexpected error handling REGISTER_TOOL: {e}"
        print(error_msg)
        messager.log(error_msg, level="error")
        import traceback

        traceback.print_exc()
