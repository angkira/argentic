import json
import sys
import yaml
import threading
import time  # Added for sleep
import uuid  # For generating a unique session user_id

# Local application imports
from core.messager import Messager, MQTTMessage
from core.decorators import mqtt_handler_decorator  # Import the decorator

# --- Load Configuration ---
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

# --- Configuration Values ---
MQTT_BROKER = config["mqtt"]["broker_address"]
MQTT_PORT = config["mqtt"]["port"]
MQTT_KEEPALIVE = config["mqtt"]["keepalive"]
MQTT_CLIENT_ID = config["mqtt"].get(
    "cli_client_id", "rag_cli_client"
)  # Use different client ID

# Get topic for 'ask_question' handler
ASK_TOPIC = None
for topic, handler in config.get("mqtt", {}).get("subscriptions", {}).items():
    if handler == "handle_ask_question":
        ASK_TOPIC = topic
        break
if ASK_TOPIC is None:
    print("CLI Error: 'ask_question' topic not found in configuration.")
    sys.exit(1)
MQTT_TOPIC_ASK = ASK_TOPIC

MQTT_TOPIC_ANSWER = config["mqtt"]["publish_topics"]["response"]
MQTT_PUB_LOG = config["mqtt"]["publish_topics"]["log"]  # Needed for Messager


# --- Response Handler ---
# Instantiate Messager
messager = Messager(
    broker_address=MQTT_BROKER,
    port=MQTT_PORT,
    client_id=MQTT_CLIENT_ID,
    keepalive=MQTT_KEEPALIVE,
    pub_log_topic=MQTT_PUB_LOG,
)

# Create decorator instance with the messager
handler_decorator = mqtt_handler_decorator(messager)


@handler_decorator
# Change signature to accept parsed data and original message
def handle_answer(data: dict, msg: MQTTMessage) -> None:
    """Handles incoming answer messages."""
    question = data.get("question", "N/A")
    answer = data.get("answer")
    error = data.get("error")

    print("\n--- Agent Response ---")
    print(f"Question: {question}")
    if answer:
        print(f"Answer: {answer}")
    elif error:
        print(f"Error: {error}")
    else:
        print(f"Received unknown format: {data}")
    print("----------------------")
    print("> ", end="", flush=True)  # Re-print the input prompt


# --- Main CLI Loop ---
if __name__ == "__main__":
    # Generate a persistent user_id for this CLI session
    USER_ID = str(uuid.uuid4())
    print(f"CLI: Using session user_id={USER_ID}")
    # Define subscriptions and handlers for the CLI
    # Subscribe to the exact response topic
    cli_subscriptions = {MQTT_TOPIC_ANSWER: "handle_answer"}
    cli_handlers = {"handle_answer": handle_answer}

    try:
        print("CLI: Connecting...")
        # Connect and wait
        if not messager.connect():
            print("CLI Error: Initial connection failed. Exiting.")
            sys.exit(1)

        if not messager._connection_event.wait(timeout=10.0):  # Wait for connection
            print("CLI Error: Connection timeout. Exiting.")
            messager.disconnect()  # Clean up
            sys.exit(1)

        # Register handlers *after* connection confirmed
        for topic, handler_name in cli_subscriptions.items():
            handler_func = cli_handlers.get(handler_name)
            if handler_func:
                messager.register_handler(topic, handler_func)
        # Debug: show CLI subscriptions
        print(f"CLI: Subscribed to topics: {list(messager._topic_handlers.keys())}")

        # Start background loop for receiving messages
        messager.start_background_loop()

        print("\n--- RAG CLI Client ---")
        print("Type your question and press Enter.")
        print("Type 'quit' or 'exit' to leave.")

        while True:
            # Check connection status periodically
            if not messager.is_connected():
                print("\nCLI Error: Disconnected. Please restart the client.")
                # Simple exit on disconnect for the client
                break

            user_input = input("> ")
            if user_input.lower() in ["quit", "exit"]:
                break

            if not user_input:
                continue

            print("\nRequesting answer...")
            # Construct payload including session user_id and question
            payload = {"question": user_input, "user_id": USER_ID}
            messager.publish(MQTT_TOPIC_ASK, payload)

    except KeyboardInterrupt:
        print("\nCLI: Exiting...")
    except Exception as e:
        print(f"\nCLI Error: An unexpected error occurred: {e}")
    finally:
        print("CLI: Shutting down...")
        messager.stop()  # Gracefully stop the messager (disconnects and stops loop)
        print("CLI: Shutdown complete.")
