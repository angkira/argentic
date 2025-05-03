import time
from typing import Any, Dict, Optional

from core.messager import Messager, MQTTMessage
from core.agent import Agent


def handle_ask_question(
    messager: Messager,
    agent: Optional[Agent],
    data: Optional[Dict[str, Any]],
    msg: MQTTMessage,
    handler_kwargs: Dict[str, Any],
) -> None:
    """Handles messages on the 'ask_question' topic (core logic)."""
    topic = msg.topic
    pub_response_topic = handler_kwargs.get("pub_response_topic")

    if agent is None:
        err_msg = f"Agent not available for handle_ask_question on topic {topic}"
        print(err_msg)
        messager.publish_log(err_msg, level="error")
        if pub_response_topic:
            messager.publish(
                pub_response_topic, {"error": "Internal configuration error: Agent missing"}
            )
        return

    if pub_response_topic is None:
        messager.publish_log(f"Response topic not configured for {topic}", level="error")
        # Cannot publish response, maybe just log the question processing
        # return # Or continue processing but without publishing result?

    # Handle potential non-JSON payload
    if data is None:
        # Log handled by decorator
        if pub_response_topic:
            messager.publish(pub_response_topic, {"error": "Invalid or missing JSON payload"})
        return

    question: Optional[str] = data.get("question")
    if question:
        user_id = data.get("user_id")  # may be None
        collection_name = data.get("collection")  # Optional collection targeting
        print(f"\n🤔 Processing Question via MQTT (user={user_id}): {question}")
        start_time = time.time()
        # Invoke Agent with question
        raw_resp = agent.query(question, collection_name=collection_name, user_id=user_id)
        response_str = str(raw_resp)
        end_time = time.time()
        print(f"\n💡 Answer: {response_str}")

        # Publish response only if topic is configured
        if pub_response_topic:
            if "Sorry, an error occurred" in response_str:
                messager.publish(
                    pub_response_topic,
                    {"user_id": user_id, "question": question, "error": response_str},
                )
            else:
                messager.publish(
                    pub_response_topic,
                    {"user_id": user_id, "question": question, "answer": response_str},
                )

        print(f"(Response time: {end_time - start_time:.2f} seconds)")
        messager.publish_log(f"Processed question '{question}' in {end_time - start_time:.2f}s.")
    else:
        messager.publish_log(
            f"Received ask_question command on {topic} with missing 'question'.",
            level="warning",
        )
        if pub_response_topic:
            messager.publish(pub_response_topic, {"error": "Missing 'question' in payload"})
