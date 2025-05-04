import time
from typing import Any, Dict, Optional

# Core components
from core.messager import Messager, MQTTMessage
from core.agent import Agent  # Agent might hold status info

# Import the specific Pydantic message types
from core.protocol.message import StatusRequestMessage, StatusMessage


# Decorator is applied in main.py


def handle_status_request(
    messager: Messager,
    agent: Agent,  # Inject Agent
    message: StatusRequestMessage,  # Inject the parsed Pydantic message object
    mqtt_msg: MQTTMessage,
    handler_kwargs: Dict[str, Any],
) -> None:
    """Handles requests for system status information."""
    topic = mqtt_msg.topic
    messager.log(
        f"Handler '{handle_status_request.__name__}': Received status request on {topic} from {message.source}"
    )

    try:
        # Gather status information from relevant components
        status_data = {
            "service_id": messager.client_id,
            "mqtt_connected": messager.is_connected(),
            "registered_tools": list(agent.tool_manager.tools.keys()),
            "llm_backend": agent.llm.backend_name if agent.llm else "N/A",
            # Access values from handler_kwargs instead of config
            "llm_model": handler_kwargs.get("llm_model", "unknown"),
            "embedding_model": handler_kwargs.get("embedding_model", "unknown"),
            "default_collection": handler_kwargs.get("default_collection_name", "unknown"),
            "mqtt_broker": handler_kwargs.get("mqtt_broker", "unknown"),
            "subscribed_topics": handler_kwargs.get("subscribed_topics", []),
            "request_details_echo": message.request_details,  # Echo back any details from request
        }

        # Create and publish the StatusMessage response
        response_topic = handler_kwargs.get("pub_status_topic", "agent/status/info")
        response_message = StatusMessage(
            source=messager.client_id,
            data=status_data,
            recipient=message.source,  # Send back to the original requester
        )
        messager.publish(response_topic, response_message.model_dump_json())
        messager.log(
            f"Handler '{handle_status_request.__name__}': Sent status response to {message.source}"
        )

    except Exception as e:
        error_msg = f"Error processing status request from {message.source}: {e}"
        messager.log(f"Handler '{handle_status_request.__name__}': {error_msg}", level="error")
        # Optional: Send an error status message back
        error_topic = handler_kwargs.get("pub_error_topic", "agent/status/error")
        # error_resp = ErrorMessage(
        #     source=messager.client_id,
        #     data={"status": "error", "action": "status_request", "details": error_msg},
        #     recipient=message.source
        # )
        # messager.publish(error_topic, error_resp.model_dump_json())
