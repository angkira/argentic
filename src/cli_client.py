import sys
import yaml
import uuid
import json

from core.messager import Messager, MQTTMessage
from core.decorators import mqtt_handler_decorator
from core.protocol.message import AskQuestionMessage, AnswerMessage, from_mqtt_message, AnyMessage
from typing import Any, Dict, Optional

CONFIG_PATH = "config.yaml"
try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    print(f"CLI: Configuration loaded from '{CONFIG_PATH}'.")
except FileNotFoundError:
    print(f"CLI Error: Configuration file '{CONFIG_PATH}' not found.")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"CLI Error: Parsing configuration file '{CONFIG_PATH}': {e}")
    sys.exit(1)

# --- MQTT Configuration ---
MQTT_BROKER = config["mqtt"]["broker_address"]
MQTT_PORT = config["mqtt"]["port"]
MQTT_KEEPALIVE = config["mqtt"]["keepalive"]
MQTT_CLIENT_ID = config["mqtt"].get(
    "cli_client_id", f"cli_client_{uuid.uuid4()}"
)  # Ensure unique ID

# Find ASK topic based on handler name in config
ASK_TOPIC = None
for topic, handler in config.get("mqtt", {}).get("subscriptions", {}).items():
    if handler == "handle_ask_question":
        ASK_TOPIC = topic
        break
if ASK_TOPIC is None:
    print("CLI Error: Topic for 'handle_ask_question' not found in configuration.")
    sys.exit(1)
MQTT_TOPIC_ASK = ASK_TOPIC

# Get other relevant topics
MQTT_TOPIC_ANSWER = config["mqtt"]["publish_topics"]["response"]
MQTT_PUB_LOG = config["mqtt"]["publish_topics"]["log"]  # For messager internal logging

# --- Initialize Messager ---
messager = Messager(
    broker_address=MQTT_BROKER,
    port=MQTT_PORT,
    client_id=MQTT_CLIENT_ID,
    keepalive=MQTT_KEEPALIVE,
    pub_log_topic=MQTT_PUB_LOG,
)


# --- Define MQTT Handlers ---
# Define RAW handlers - update signature to expect parsed message
def handle_answer(messager: Messager, message: AnswerMessage, mqtt_msg: MQTTMessage) -> None:
    """Handles incoming AnswerMessages on the answer topic."""
    # Access data directly from the Pydantic model attributes
    print("\n--- Agent Response ---")
    print(f"Question: {message.question}")  # Echoed question
    if message.answer:
        # Clean up the answer - strip markdown code fences if present
        answer = message.answer
        if "```" in answer:
            import re

            # Find and extract content between markdown code fences
            code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
            code_blocks = re.findall(code_block_pattern, answer)
            if code_blocks:
                # If there are code blocks, combine them
                answer = "\n".join(code_blocks)
            else:
                # If regex didn't find anything but ``` exists, do basic cleanup
                answer = answer.replace("```json", "").replace("```", "").strip()

        print(f"Answer: {answer}")
    elif message.error:
        print(f"Error: {message.error}")
    else:
        # Fallback if format is unexpected (less likely with Pydantic)
        print(f"Received unexpected AnswerMessage format: {message.model_dump_json(indent=2)}")
    print("----------------------")
    # Re-display prompt for next input
    print("> ", end="", flush=True)


def handle_status(messager: Messager, message: AnyMessage, mqtt_msg: MQTTMessage) -> None:
    """Handles incoming messages on the status topic."""
    print("\n--- Status Update ---")
    # Print the Pydantic model as JSON
    print(message.model_dump_json(indent=2))
    print("----------------------")
    print("> ", end="", flush=True)


# --- Main Execution Block ---
if __name__ == "__main__":
    USER_ID = str(uuid.uuid4())  # Unique ID for this CLI session
    print(f"CLI: Using session user_id={USER_ID}")

    # Define subscriptions and handlers for this client
    cli_subscriptions = {
        MQTT_TOPIC_ANSWER: "handle_answer",
    }
    cli_handlers = {
        "handle_answer": handle_answer,
    }

    try:
        print("CLI: Connecting to MQTT broker...")
        if not messager.connect():
            print("CLI Error: Initial connection failed. Exiting.")
            sys.exit(1)

        # Wait for connection confirmation
        if not messager._connection_event.wait(timeout=10.0):
            print("CLI Error: Connection timeout. Exiting.")
            messager.disconnect()
            sys.exit(1)
        print(f"CLI: Connected to {MQTT_BROKER}:{MQTT_PORT}")

        # Register handlers for subscribed topics
        print("CLI: Registering handlers...")
        for topic, handler_name in cli_subscriptions.items():
            raw_handler_func = cli_handlers.get(handler_name)
            if raw_handler_func:
                # Apply the decorator manually here
                # The decorator now handles parsing based on the handler's type hint
                decorated_handler = mqtt_handler_decorator(messager=messager)(raw_handler_func)
                messager.register_handler(topic, decorated_handler)
                print(f"  - Registered handler for topic: {topic}")
            else:
                print(f"  - Warning: Handler function '{handler_name}' not found.")

        print(f"CLI: Subscribed to topics: {list(messager._topic_handlers.keys())}")

        # Start the MQTT background loop for receiving messages
        messager.start_background_loop()

        print("\n--- Agent CLI Client ---")
        print("Type your question and press Enter.")
        print("Type 'quit' or 'exit' to leave.")

        # --- Input Loop ---
        while True:
            # Check connection status before prompting
            if not messager.is_connected():
                print("\nCLI Error: Disconnected from MQTT broker. Please restart the client.")
                break

            try:
                user_input = input("> ")
            except EOFError:  # Handle Ctrl+D
                print("\nCLI: EOF received, exiting...")
                break

            if user_input.lower() in ["quit", "exit"]:
                break

            if not user_input.strip():  # Ignore empty input
                continue

            # --- Publish Question using AskQuestionMessage ---
            print("\nRequesting answer from agent...")
            # Create the Pydantic message object
            ask_message = AskQuestionMessage(
                question=user_input,
                user_id=USER_ID,
                source=MQTT_CLIENT_ID,  # Identify the CLI client as source
            )
            # Publish the serialized Pydantic model
            messager.publish(MQTT_TOPIC_ASK, ask_message.model_dump_json())
            # Response handled asynchronously by handle_answer

    except KeyboardInterrupt:
        print("\nCLI: Ctrl+C received, exiting...")
    except Exception as e:
        print(f"\nCLI Error: An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()  # Print stack trace for debugging
    finally:
        print("CLI: Shutting down...")
        messager.stop()  # Disconnect MQTT and stop background loop
        print("CLI: Shutdown complete.")
