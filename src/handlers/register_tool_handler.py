import uuid
from typing import Dict, Any

# Local application imports
from core.messager import Messager, MQTTMessage

# Import Agent instead of RAGController/RAGManager
from core.agent import Agent
from core.protocol.message import ToolRegisteredMessage


# Note: The mqtt_handler_decorator will be applied in main.py
def handle_register_tool(
    messager: Messager,
    agent: Agent,  # Changed rag_controller to agent
    data: Dict[str, Any],  # Deserialized message data
    msg: MQTTMessage,  # Raw MQTT message
    handler_kwargs: Dict[str, Any],  # Additional kwargs from decorator
) -> None:
    """Handles incoming tool registration requests."""  # Removed backslash
    try:
        # The decorator should have already deserialized based on type,
        # but we can double-check or directly use the expected fields.
        # We expect 'data' to contain keys from RegisterToolMessage.to_dict()
        tool_name = data.get("tool_name")
        tool_manual = data.get("tool_manual")
        tool_api = data.get("tool_api")
        source = data.get("source")  # Get the original source from the message data

        if not all([tool_name, tool_manual, tool_api, source]):
            raise ValueError(
                "Missing required fields (tool_name, tool_manual, tool_api, source) in registration message data."
            )

        # Generate a unique ID for the tool
        tool_id = str(uuid.uuid4())

        # Register the tool with the Agent
        tool_description = agent.register_tool(  # Changed rag_controller to agent
            tool_id=tool_id,
            tool_name=tool_name,
            tool_manual=tool_manual,
            tool_api=tool_api,
        )

        messager.log(
            f"Registered tool '{tool_name}' with ID: {tool_id}. Tool description generated."
        )
        print(f"Registered tool '{tool_name}' (ID: {tool_id}) from source '{source}'.")
        print(f"Tool Description:\n{tool_description}")  # Log locally for now

        # Create and send the TOOL_REGISTERED confirmation message
        # Send it back to the original source of the registration request
        confirmation_message = ToolRegisteredMessage(
            source=handler_kwargs.get("mqtt_client_id", "rag_agent"),  # Source is the agent itself
            tool_id=tool_id,
            tool_name=tool_name,
            recipient=source,  # Send back to the original requester
        )

        # Determine the response topic - perhaps a dedicated topic or back to the source?
        # For now, let's assume a general status topic or a specific tool registration confirmation topic.
        # Using the general status topic defined in main.py for now.
        response_topic = handler_kwargs.get(
            "pub_status_topic", "rag/status/info"
        )  # Default if not passed

        messager.publish(response_topic, confirmation_message.to_dict())
        messager.log(
            f"Sent TOOL_REGISTERED confirmation for tool '{tool_name}' to topic '{response_topic}' for recipient '{source}'."
        )

    except ValueError as ve:
        error_msg = f"Error processing REGISTER_TOOL message: {ve}"
        print(error_msg)
        messager.log(error_msg, level="error")
        # Optionally send an ERROR message back
    except Exception as e:
        error_msg = f"Unexpected error handling REGISTER_TOOL: {e}"
        print(error_msg)
        messager.log(error_msg, level="error")
        # Optionally send an ERROR message back
