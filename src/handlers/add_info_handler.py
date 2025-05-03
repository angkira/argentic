from typing import Any, Dict, Optional

# Note: Imports remain the same relative to the project root
from core.messager import Messager, MQTTMessage
from core.rag import RAGController

# Decorator is no longer applied here, but imported for type hinting if needed
# from core.decorators import mqtt_handler_decorator


# Raw handler function - decorator will be applied in main.py
def handle_add_info(
    messager: Messager,
    rag_controller: Optional[RAGController],
    data: Optional[Dict[str, Any]],
    msg: MQTTMessage,
    handler_kwargs: Dict[str, Any],
) -> None:
    """Handles messages on the 'add_info' topic (core logic)."""
    topic = msg.topic
    pub_status_topic = handler_kwargs.get("pub_status_topic")

    # Ensure RAGController was injected if needed for this handler
    if rag_controller is None:
        err_msg = f"RAGController not available for handle_add_info on topic {topic}"
        print(err_msg)
        messager.publish_log(err_msg, level="error")
        if pub_status_topic:
            messager.publish(
                pub_status_topic,
                {
                    "status": "error",
                    "topic": topic,
                    "error": "Internal configuration error: RAGController missing",
                },
            )
        return

    # Handle potential non-JSON payload (checked by decorator, data will be None)
    if data is None:
        # Log is handled by decorator, but we might want specific status publishing
        if pub_status_topic:
            messager.publish(
                pub_status_topic,
                {
                    "status": "error",
                    "topic": topic,
                    "error": "Invalid or missing JSON payload",
                },
            )
        return  # No further processing possible without data

    text: Optional[str] = data.get("text")
    source: str = data.get("source", "mqtt_input")  # Default source if not provided
    timestamp: Optional[float] = data.get("timestamp")

    if text:
        success: bool = rag_controller.remember(text, source, timestamp)
        status = "processed" if success else "error_remembering"
        if pub_status_topic:
            messager.publish(
                pub_status_topic,
                {"status": status, "topic": topic, "source": source},
            )
        else:
            messager.publish_log(f"Status topic not configured for {topic}", level="warning")

    else:
        messager.publish_log(
            f"Received add_info command on {topic} with missing 'text'.", level="warning"
        )
        if pub_status_topic:
            messager.publish(
                pub_status_topic,
                {
                    "status": "error",
                    "topic": topic,
                    "error": "Missing 'text' in payload",
                },
            )
