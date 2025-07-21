import asyncio
import os
import yaml
from dotenv import load_dotenv  # Import load_dotenv

from argentic.core.agent.agent import Agent
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider  # Updated import
from argentic.core.messager.messager import Messager


async def main():
    # Load environment variables from .env file
    load_dotenv()

    # Load LLM configuration from a minimal config file
    llm_config_path = os.path.join(
        os.path.dirname(__file__), "config_gemini.yaml"
    )  # Updated config file
    with open(llm_config_path, "r") as f:
        llm_full_config = yaml.safe_load(f)
    if llm_full_config is None:
        llm_full_config = {}
    llm_config = llm_full_config.get("llm", {})
    if not isinstance(llm_config, dict):
        print(
            "Warning: 'llm' configuration not found or not a dictionary in config_gemini.yaml. Using defaults."
        )
        llm_config = {}

    # Load Messaging configuration from a minimal config file
    messaging_config_path = os.path.join(os.path.dirname(__file__), "config_messaging.yaml")
    with open(messaging_config_path, "r") as f:
        messaging_full_config = yaml.safe_load(f)
    if messaging_full_config is None:
        messaging_full_config = {}
    messaging_config_data = messaging_full_config.get("messaging", {})
    if not isinstance(messaging_config_data, dict):
        print(
            "Warning: 'messaging' configuration not found or not a dictionary in config_messaging.yaml. Using defaults."
        )
        messaging_config_data = {}

    # Initialize LLM (Google Gemini) using config (api_key will be read internally)
    llm = GoogleGeminiProvider(config=llm_config)  # Corrected instantiation

    # Initialize Messager using config
    broker_address = messaging_config_data.get("broker_address", "localhost")
    port = messaging_config_data.get("port", 1883)
    keepalive = messaging_config_data.get("keepalive", 60)
    messager = Messager(
        broker_address=broker_address,
        port=port,
        keepalive=keepalive,
    )
    await messager.connect()

    # Create a single agent
    agent = Agent(llm=llm, messager=messager, role="default_agent")
    await agent.async_init()

    print("\n--- Running Single Agent Example ---")
    question = "What is the capital of France?"
    print(f"Question: {question}")
    answer = await agent.query(question)
    print(f"Answer: {answer}")

    question = "Tell me a short story about a brave knight and a dragon."
    print(f"\nQuestion: {question}")
    answer = await agent.query(question)
    print(f"Answer: {answer}")

    await messager.disconnect()


if __name__ == "__main__":
    # We call load_dotenv() at the beginning of main()
    asyncio.run(main())
