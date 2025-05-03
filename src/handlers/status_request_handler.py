import time
from typing import Any, Dict, Optional

from core.messager import Messager, MQTTMessage


def handle_status_request(
    messager: Messager,
    data: Optional[Dict[str, Any]],
    msg: MQTTMessage,
    handler_kwargs: Dict[str, Any],
) -> None:
    """Handles messages on the 'status_request' topic (core logic)."""
    topic = msg.topic
    pub_status_topic = handler_kwargs.get("pub_status_topic")
    llm_model = handler_kwargs.get("llm_model", "Unknown")
    embedding_model = handler_kwargs.get("embedding_model", "Unknown")
    default_collection_name = handler_kwargs.get("default_collection_name", "Unknown")
    mqtt_broker = handler_kwargs.get("mqtt_broker", "Unknown")
    subscribed_topics = handler_kwargs.get("subscribed_topics", [])

    messager.publish_log(f"Processing status request received on topic {topic}")

    status_info: Dict[str, Any] = {
        "status": "running",
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "default_collection": default_collection_name,
        "mqtt_broker": mqtt_broker,
        "subscribed_topics": subscribed_topics,
        "timestamp": time.time(),
    }

    if pub_status_topic:
        messager.publish(pub_status_topic, status_info)
    else:
        messager.publish_log(f"Status topic not configured for {topic}", level="error")
