import sys
import yaml
import uuid
import json
from typing import Any, Dict, Optional

from core.client import Client
from core.messager import Messager, MQTTMessage
from core.decorators import mqtt_handler_decorator
from core.protocol.message import AskQuestionMessage, AnswerMessage, from_mqtt_message, AnyMessage
from core.logger import LogLevel

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


class CliClient(Client):
    """CLI Client implementation of the base Client class"""

    def __init__(self):
        """Initialize the CLI Client"""
        # Create the messager instance
        messager = Messager(
            broker_address=MQTT_BROKER,
            port=MQTT_PORT,
            client_id=MQTT_CLIENT_ID,
            keepalive=MQTT_KEEPALIVE,
            pub_log_topic=MQTT_PUB_LOG,
        )

        # Define subscriptions and handlers for this client
        subscriptions = {
            MQTT_TOPIC_ANSWER: "handle_answer",
        }

        handlers = {
            "handle_answer": self.handle_answer,
        }

        # Initialize the base Client
        super().__init__(
            messager=messager,
            client_id=MQTT_CLIENT_ID,
            subscriptions=subscriptions,
            handlers=handlers,
            log_level=LogLevel.INFO,
        )

        self.ask_topic = MQTT_TOPIC_ASK

    def handle_answer(
        self, messager: Messager, message: AnswerMessage, mqtt_msg: MQTTMessage
    ) -> None:
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

    def run_interactive(self):
        """Run the CLI client in interactive mode"""
        try:
            print(f"CLI: Using session user_id={self.user_id}")

            if not self.start():
                print("CLI Error: Failed to start client. Exiting.")
                return

            print("\n--- Agent CLI Client ---")
            print("Type your question and press Enter.")
            print("Type 'quit' or 'exit' to leave.")

            # --- Input Loop ---
            while True:
                # Check connection status before prompting
                if not self.messager.is_connected():
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

                # Ask question using the base Client method
                self.ask_question(user_input, self.ask_topic)
                # Response handled asynchronously by handle_answer

        except KeyboardInterrupt:
            print("\nCLI: Ctrl+C received, exiting...")
        except Exception as e:
            print(f"\nCLI Error: An unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()  # Print stack trace for debugging
        finally:
            print("CLI: Shutting down...")
            self.stop()  # Use the base Client's stop method
            print("CLI: Shutdown complete.")


# --- Main Execution Block ---
if __name__ == "__main__":
    cli_client = CliClient()
    cli_client.run_interactive()
