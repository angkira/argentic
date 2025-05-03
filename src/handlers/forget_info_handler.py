from typing import Any, Dict, Optional

from core.messager import Messager, MQTTMessage
from core.rag import RAGManager


def handle_forget_info(
    messager: Messager,
    rag_manager: Optional[RAGManager],
    data: Optional[Dict[str, Any]],
    msg: MQTTMessage,
    handler_kwargs: Dict[str, Any],
) -> None:
    """Handles messages on the 'forget_info' topic (core logic)."""
    topic = msg.topic
    pub_status_topic = handler_kwargs.get("pub_status_topic")

    if rag_manager is None:
        err_msg = f"RAGManager not available for handle_forget_info on topic {topic}"
        print(err_msg)
        messager.publish_log(err_msg, level="error")
        if pub_status_topic:
            messager.publish(
                pub_status_topic,
                {
                    "status": "error",
                    "topic": topic,
                    "error": "Internal configuration error: RAGManager missing",
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
    collection_name: Optional[str] = data.get("collection")  # Optional collection targeting

    if where_filter and isinstance(where_filter, dict):
        result: Dict[str, Any] = rag_manager.forget(where_filter, collection_name)
        if pub_status_topic:
            response_payload = {
                "status": result["status"],
                "topic": topic,
                "filter": where_filter,
                "deleted_count": result["deleted_count"],
                "message": result.get("message"),  # Include message if present
            }
            if collection_name:
                response_payload["collection"] = collection_name

            messager.publish(pub_status_topic, response_payload)
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
