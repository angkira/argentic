import time
import json
import re
import traceback
from typing import Any, Dict, Optional, List

from core.messager import Messager
from core.agent import Agent
from core.protocol.message import AskQuestionMessage, AnswerMessage, InfoMessage
from core.logger import get_logger, LogLevel


# Create a handler-specific logger
logger = get_logger("ask_handler")


def extract_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extract tool calls from JSON in the text.
    Returns a list of tool calls if found, None otherwise.
    """
    # Try to find JSON inside markdown code blocks first
    if "```" in text:
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        code_blocks = re.findall(code_block_pattern, text)
        if code_blocks:
            text = code_blocks[0].strip()

    # Now try to parse the possibly JSON text
    try:
        # Check if the text is valid JSON and contains tool_calls
        data = json.loads(text)
        if isinstance(data, dict) and "tool_calls" in data and isinstance(data["tool_calls"], list):
            return data["tool_calls"]
    except json.JSONDecodeError:
        # Try to extract JSON object using regex as fallback
        json_pattern = r"\{[\s\S]*\}"
        match = re.search(json_pattern, text)
        if match:
            try:
                data = json.loads(match.group(0))
                if (
                    isinstance(data, dict)
                    and "tool_calls" in data
                    and isinstance(data["tool_calls"], list)
                ):
                    return data["tool_calls"]
            except json.JSONDecodeError:
                pass
    return None


def handle_ask_question(
    messager: Messager,
    agent: Agent,  # Agent injected by decorator
    message: AskQuestionMessage,  # Inject the parsed Pydantic message object
    mqtt_msg: Any,  # Inject original MQTT message if needed (e.g., for topic)
    handler_kwargs: Dict[str, Any],  # Inject decorator kwargs if needed
) -> None:
    """Handles 'ask_question' requests using the Agent."""
    topic = mqtt_msg.topic

    # Use both local logger and MQTT logging
    logger.info(f"Received question from {message.source} on {topic}: '{message.question}'")
    messager.log(
        f"Handler '{handle_ask_question.__name__}': Received question from {message.source} on {topic}: '{message.question}'"
    )

    # Get response topic from handler_kwargs
    response_topic = handler_kwargs.get("pub_response_topic", "agent/response/answer")
    status_topic = handler_kwargs.get("pub_status_topic", "agent/status/info")

    try:
        # First pass: Get initial response from agent
        logger.info(f"Querying agent for question: '{message.question[:50]}...'")
        initial_response = agent.query(
            question=message.question,
            collection_name=message.collection_name,
            user_id=message.user_id,
        )

        # Check if the response contains tool calls
        tool_calls = extract_tool_calls(initial_response)

        if tool_calls:
            # Send a status message to client that we're processing with tools
            status_message = InfoMessage(
                source=messager.client_id,
                data={
                    "status": "processing",
                    "message": "Using tools to answer your question...",
                    "question": message.question,
                },
                recipient=message.source,
            )
            messager.publish(status_topic, status_message.model_dump_json())

            logger.info(f"Detected tool calls in response. Processing tools before responding.")
            messager.log(
                f"Handler: Detected tool calls in response. Processing tools before responding."
            )

            # Process each tool call
            tool_results = []
            for call in tool_calls:
                tool_id = call.get("tool_id")
                arguments = call.get("arguments")

                if tool_id and isinstance(arguments, dict):
                    # Send status update about which tool we're using
                    status_message = InfoMessage(
                        source=messager.client_id,
                        data={
                            "status": "tool_executing",
                            "tool_id": tool_id,
                            "arguments": arguments,
                            "question": message.question,
                        },
                        recipient=message.source,
                    )
                    messager.publish(status_topic, status_message.model_dump_json())

                    # Execute the tool
                    logger.info(f"Executing tool '{tool_id}' with args: {arguments}")
                    messager.log(f"Handler: Executing tool '{tool_id}' with args: {arguments}")
                    tool_result = agent.tool_manager.execute_tool(tool_id, arguments)

                    tool_results.append({"tool_id": tool_id, "result": tool_result})
                    logger.debug(f"Tool '{tool_id}' returned result: {str(tool_result)[:100]}...")

            # If we have tool results, pass them back to the agent for a final answer
            if tool_results:
                logger.info(f"Got {len(tool_results)} tool results, requesting final answer...")
                messager.log(
                    f"Handler: Got {len(tool_results)} tool results, requesting final answer..."
                )

                # Format the tool results for the follow-up question
                tool_results_str = "\n\n".join(
                    [
                        f"TOOL RESULT (from {result['tool_id']}):\n{result['result']}"
                        for result in tool_results
                    ]
                )

                follow_up_question = (
                    f'I\'ve used tools to help answer the question: "{message.question}"\n\n'
                    f"Here are the tool results:\n\n{tool_results_str}\n\n"
                    f"Based on these results, please provide a final comprehensive answer to the original question."
                )

                # Get the final answer from the agent
                logger.info("Requesting final answer from agent with tool results")
                final_answer = agent.query(
                    question=follow_up_question,
                    collection_name=message.collection_name,
                    user_id=message.user_id,
                )

                # Check the final answer doesn't also contain tool calls
                if extract_tool_calls(final_answer):
                    logger.warning("Final answer still contains tool calls. Sending anyway.")
                    messager.log(
                        "Handler: Warning - Final answer still contains tool calls. Sending anyway.",
                        level="warning",
                    )

                # Now send the final answer to the client
                response_message = AnswerMessage(
                    source=messager.client_id,
                    question=message.question,
                    answer=final_answer,
                    recipient=message.source,
                    user_id=message.user_id,
                )
                messager.publish(response_topic, response_message.model_dump_json())

                logger.info(f"Sent final answer after tool processing to {response_topic}")
                messager.log(f"Handler: Sent final answer after tool processing.")
                return

        # If no tool calls or after processing, send the response
        response_message = AnswerMessage(
            source=messager.client_id,
            question=message.question,
            answer=initial_response,
            recipient=message.source,
            user_id=message.user_id,
        )
        messager.publish(response_topic, response_message.model_dump_json())

        logger.info(f"Sent direct answer for '{message.question[:50]}...' to {response_topic}")
        messager.log(
            f"Handler '{handle_ask_question.__name__}': Sent answer for '{message.question[:50]}...' to {response_topic}"
        )

    except Exception as e:
        error_msg = f"Error processing question '{message.question[:50]}...': {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())  # Log full traceback
        messager.log(f"Handler '{handle_ask_question.__name__}': {error_msg}", level="error")

        # Send an error response back
        try:
            error_response = AnswerMessage(
                source=messager.client_id,
                question=message.question,
                error=error_msg,
                recipient=message.source,
                user_id=message.user_id,
            )
            messager.publish(response_topic, error_response.model_dump_json())
            logger.info("Sent error response to client")
        except Exception as pub_e:
            logger.error(f"Failed to publish error response: {pub_e}")
            logger.error(traceback.format_exc())  # Log full traceback
            messager.log(
                f"Handler '{handle_ask_question.__name__}': Failed to publish error response: {pub_e}",
                level="error",
            )
