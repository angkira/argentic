import asyncio
import sys
from typing import Optional, Any, Dict
import yaml
import uuid
import concurrent.futures

from core.client import Client
from core.messager.messager import Messager
from core.protocol.message import (
    BaseMessage,
    AnswerMessage,
    AskQuestionMessage,
    AgentLLMResponseMessage,
)
from core.protocol.task import TaskResultMessage, TaskErrorMessage
from core.logger import LogLevel, get_logger

CONFIG_PATH = "config.yaml"
config: Dict[str, Any] = {}
try:
    with open(CONFIG_PATH, "r") as f:
        loaded_config = yaml.safe_load(f)
        if isinstance(loaded_config, dict):
            config = loaded_config
        else:
            print(f"CLI Error: Configuration in '{CONFIG_PATH}' is not a valid dictionary.")
            sys.exit(1)
    print(f"CLI: Configuration loaded from '{CONFIG_PATH}'.")
except FileNotFoundError:
    print(f"CLI Error: Configuration file '{CONFIG_PATH}' not found.")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"CLI Error: Parsing configuration file '{CONFIG_PATH}': {e}")
    sys.exit(1)


# --- MESSAGING Configuration with robust access ---
def get_config_value(cfg_dict, path, default=None, required=True):
    keys = path.split(".")
    val = cfg_dict
    for key in keys:
        if isinstance(val, dict) and key in val:
            val = val[key]
        else:
            if required:
                print(f"CLI Error: Missing required config: '{path}' in '{CONFIG_PATH}'.")
                sys.exit(1)
            return default
    return val


messaging_config = get_config_value(config, "messaging", {}, required=True)
MESSAGING_BROKER = get_config_value(messaging_config, "broker_address", required=True)
MESSAGING_PORT = get_config_value(messaging_config, "port", 1883, required=False)
MESSAGING_KEEPALIVE = get_config_value(messaging_config, "keepalive", 60, required=False)
MESSAGING_CLIENT_ID = messaging_config.get("cli_client_id", f"cli_client_{uuid.uuid4()}")

MESSAGING_TOPIC_ASK = get_config_value(config, "topics.commands.ask_question", required=True)
MESSAGING_TOPIC_ANSWER = get_config_value(config, "topics.responses.answer", required=True)
MESSAGING_PUB_LOG = get_config_value(config, "topics.log", required=False)

MESSAGING_TOPIC_AGENT_LLM_RESPONSE = get_config_value(
    config, "topics.agent_events.llm_response", required=False
)
MESSAGING_TOPIC_AGENT_TOOL_RESULT = get_config_value(
    config, "topics.agent_events.tool_result", required=False
)

logger = get_logger("CliClient", LogLevel.INFO)


class CliClient(Client):
    def __init__(self):
        self.client_id = MESSAGING_CLIENT_ID
        self.ask_topic = MESSAGING_TOPIC_ASK
        self.messager: Optional[Messager] = None
        self.answer_received_event: Optional[asyncio.Event] = None

    async def initialize(self):
        """Initialize the client in async context"""
        self.messager = Messager(
            protocol=get_config_value(messaging_config, "protocol", "mqtt", required=False),
            broker_address=MESSAGING_BROKER,
            port=MESSAGING_PORT,
            client_id=self.client_id,
            keepalive=MESSAGING_KEEPALIVE,
            pub_log_topic=MESSAGING_PUB_LOG,
            log_level=LogLevel.INFO,
        )
        super().__init__(messager=self.messager, client_id=self.client_id)
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logger

        self.answer_received_event = asyncio.Event()

    async def ask_question(self, question_text: str, topic: str, user_id: Optional[str] = None):
        current_user_id = user_id or self.client_id

        # Instantiate AskQuestionMessage based on latest Pydantic error:
        # - 'question' is a required top-level field.
        # - 'data' field (from BaseMessage) should be None.
        msg = AskQuestionMessage(
            question=question_text,
            user_id=current_user_id,
            source=self.client_id,
            data=None,  # Explicitly set data to None
        )

        if self.messager:
            await self.messager.publish(topic, msg)
            self.logger.info(
                f"Asking question: '{question_text}' on topic {topic} for user '{current_user_id}'"
            )
        else:
            self.logger.error("Messager not initialized. Cannot ask question.")

    async def handle_answer(self, message: AnswerMessage) -> None:
        print("\n--- Agent Final Answer ---")
        print(f"Question: {message.question}")
        if message.answer:
            answer = message.answer
            print(f"Answer: {answer}")
        elif message.error:
            print(f"Error: {message.error}")
        else:
            print(f"Received unexpected response format: {message.model_dump_json(indent=2)}")
        print("----------------------")
        if self.answer_received_event:
            self.answer_received_event.set()

    async def handle_agent_llm_thought(self, message: AgentLLMResponseMessage) -> None:
        print("\n--- Agent Thinking (LLM Response) ---")
        if message.parsed_type == "tool_call" and message.parsed_tool_calls:
            print("Agent plans to use tools:")
            for tc_item in message.parsed_tool_calls:
                print(f"  - Tool ID: {tc_item.tool_id}, Arguments: {tc_item.arguments}")
        elif message.parsed_type == "direct" and message.parsed_direct_content:
            print(f"Agent direct thought/response: {message.parsed_direct_content}")
        elif message.raw_content:
            print(f"Agent raw LLM output: {message.raw_content[:500]}...")
        print("-----------------------------------")

    async def handle_agent_tool_result(self, message: BaseMessage) -> None:
        print("\n--- Agent Tool Execution Result ---")
        if isinstance(message, TaskResultMessage):
            print(f"Tool '{message.tool_name}' (ID: {message.tool_id}) executed.")
            if message.result is not None:
                res_str = str(message.result)
                print(f"Result: {res_str[:500]}{'...' if len(res_str) > 500 else ''}")
            else:
                print("Result: (No content)")
        elif isinstance(message, TaskErrorMessage):
            print(f"Tool '{message.tool_name}' (ID: {message.tool_id}) failed.")
            print(f"Error: {message.error}")
        elif isinstance(message, BaseMessage):
            self.logger.warning(
                f"Received unhandled BaseMessage type on tool result topic: {type(message)}"
            )
            print(f"Received unexpected tool result message type: {type(message)}")
        print("-----------------------------------")

    async def run_interactive(self):
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="CLIInputWorker"
        )

        try:
            await self.initialize()
            if not self.messager:
                self.logger.error("CLI Error: Messager failed to initialize.")
                return False

            if not await self.messager.connect():
                self.logger.error("CLI Error: Failed to connect. Check logs for details.")
                return False

            await self.messager.subscribe(
                MESSAGING_TOPIC_ANSWER, self.handle_answer, message_cls=AnswerMessage
            )
            self.logger.info(f"Subscribed to final answer topic: {MESSAGING_TOPIC_ANSWER}")

            if MESSAGING_TOPIC_AGENT_LLM_RESPONSE:
                await self.messager.subscribe(
                    MESSAGING_TOPIC_AGENT_LLM_RESPONSE,
                    self.handle_agent_llm_thought,
                    message_cls=AgentLLMResponseMessage,
                )
                self.logger.info(
                    f"Subscribed to agent LLM response topic: {MESSAGING_TOPIC_AGENT_LLM_RESPONSE}"
                )

            if MESSAGING_TOPIC_AGENT_TOOL_RESULT:
                await self.messager.subscribe(
                    MESSAGING_TOPIC_AGENT_TOOL_RESULT,
                    self.handle_agent_tool_result,
                    message_cls=BaseMessage,
                )
                self.logger.info(
                    f"Subscribed to agent tool result topic: {MESSAGING_TOPIC_AGENT_TOOL_RESULT}"
                )

            print("--- Agent CLI Client ---")
            print("Type your question and press Enter.")
            print("Type 'quit' or 'exit' to leave.")
            print("> ", end="", flush=True)

            while True:
                user_input = await loop.run_in_executor(executor, sys.stdin.readline)
                user_input = user_input.strip()
                if user_input.lower() in ["quit", "exit"]:
                    break
                if user_input:
                    if self.answer_received_event:
                        self.answer_received_event.clear()
                    await self.ask_question(user_input, self.ask_topic)
                    self.logger.info(f"Waiting for final answer to: '{user_input}'...")
                    try:
                        if self.answer_received_event:
                            await asyncio.wait_for(self.answer_received_event.wait(), timeout=120.0)
                    except asyncio.TimeoutError:
                        print(
                            "\n--- No final answer received within timeout. Check for thinking steps. ---"
                        )
                    finally:
                        print("> ", end="", flush=True)
                else:
                    print("> ", end="", flush=True)
            return True
        except asyncio.CancelledError:
            self.logger.info("CLI: run_interactive task was cancelled.")
            raise
        except Exception as e:
            self.logger.error(
                f"CLI Error: An unexpected error occurred in run_interactive: {e}", exc_info=True
            )
            return False
        finally:
            self.logger.info("CLI: Shutting down (run_interactive's finally block)...")
            if self.messager and self.messager.is_connected():
                await self.messager.disconnect()
            self.logger.info("CLI: Shutdown complete (run_interactive's finally block).")
            self.logger.info("CLI: Shutting down stdin thread pool executor...")
            if sys.version_info >= (3, 9):
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=False)
            self.logger.info("CLI: Stdin thread pool executor shutdown initiated.")

    def start(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = False
        try:
            success = loop.run_until_complete(self.run_interactive())
        except KeyboardInterrupt:
            self.logger.info("CLI: KeyboardInterrupt caught in start.")
        except Exception as e:
            self.logger.error(f"CLI Error: An unexpected error in start: {e}", exc_info=True)
        finally:
            self.logger.info("CLI: Cleaning up event loop in start's finally block.")
            if self.messager and self.messager.is_connected() and not success:
                self.logger.info(
                    "CLI: Ensuring messager is disconnected due to failure or interrupt in start."
                )
                loop.run_until_complete(self.messager.disconnect())

            if not loop.is_closed():
                try:
                    all_tasks = asyncio.all_tasks(loop=loop)
                    tasks_to_cancel = [
                        t for t in all_tasks if t is not asyncio.current_task(loop=loop)
                    ]
                    if tasks_to_cancel:
                        self.logger.info(
                            f"CLI: Cancelling {len(tasks_to_cancel)} outstanding tasks..."
                        )
                        for task in tasks_to_cancel:
                            task.cancel()
                        loop.run_until_complete(
                            asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                        )
                        self.logger.info("CLI: Outstanding tasks gathered.")
                    if loop.is_running():
                        self.logger.info("CLI: Shutting down async generators...")
                        loop.run_until_complete(loop.shutdown_asyncgens())
                        self.logger.info("CLI: Async generators shutdown.")
                except RuntimeError as e_rt:
                    self.logger.warning(f"CLI: Runtime error during final loop cleanup: {e_rt}")
                except Exception as e_final_cleanup:
                    self.logger.error(
                        f"CLI: Error during final cleanup: {e_final_cleanup}", exc_info=True
                    )
                finally:
                    if not loop.is_closed():
                        self.logger.info("CLI: Closing event loop.")
                        loop.close()
                        self.logger.info("CLI: Event loop closed.")
        return success


if __name__ == "__main__":
    cli_client = CliClient()
    exit_code = 0 if cli_client.start() else 1
    sys.exit(exit_code)
