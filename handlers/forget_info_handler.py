from typing import Any, Dict, Optional

from core.messager import Messager, MQTTMessage
from core.rag import RAGController

# Decorator is no longer applied here
# from core.decorators import mqtt_handler_decorator

# Removed create_forget_info_handler factory function


def handle_forget_info(
    messager: Messager,
    rag_controller: Optional[RAGController],
    data: Optional[Dict[str, Any]],
    msg: MQTTMessage,
    handler_kwargs: Dict[str, Any],
) -> None:
    """Handles messages on the 'forget_info' topic (core logic)."""
    topic = msg.topic
    pub_status_topic = handler_kwargs.get("pub_status_topic")

    if rag_controller is None:
        err_msg = f"RAGController not available for handle_forget_info on topic {topic}"
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

    # Handle potential non-JSON payload
    if data is None:
        # Log handled by decorator
        if pub_status_topic:
            messager.publish(
                pub_status_topic,
                {
                    "status": "error",
                    "topic": topic,
                    "error": "Invalid or missing JSON payload for forget_info",
                },
            )
        return

    where_filter: Optional[Dict[str, Any]] = data.get("where_filter")

    if where_filter and isinstance(where_filter, dict):
        result: Dict[str, Any] = rag_controller.forget(where_filter)
        if pub_status_topic:
            messager.publish(
                pub_status_topic,
                {
                    "status": result["status"],
                    "topic": topic,
                    "filter": where_filter,
                    "deleted_count": result["deleted_count"],
                    "message": result.get("message"),  # Include message if present
                },
            )
        else:
            messager.publish_log(f"Status topic not configured for {topic}", level="warning")

    else:
        err_msg = f"Received forget_info command on {topic} requires a valid JSON object in 'where_filter'."
        messager.publish_log(err_msg, level="warning")
        if pub_status_topic:
            messager.publish(
                pub_status_topic,
                {"status": "error", "topic": topic, "error": err_msg},
            )
