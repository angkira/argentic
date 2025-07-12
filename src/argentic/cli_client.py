import asyncio
import sys
import signal
import select
import tty
import termios
from typing import Optional, Any, Dict
import yaml
import uuid

from argentic.core.client import Client
from argentic.core.messager.messager import Messager
from argentic.core.protocol.message import (
    BaseMessage,
    AnswerMessage,
    AskQuestionMessage,
    AgentLLMResponseMessage,
)
from argentic.core.protocol.task import TaskResultMessage, TaskErrorMessage
from argentic.core.logger import LogLevel, get_logger

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
        self._shutdown_flag = asyncio.Event()
        self._input_task: Optional[asyncio.Task] = None
        self._cleanup_started = False
        self.answer_received_event: Optional[asyncio.Event] = None
        self._original_tty_settings: Optional[Any] = None
        self.user_answer_topic: str = ""  # Will be set in initialize()

        # Will be properly initialized in initialize()
        self.messager: Optional[Messager] = None
        self.logger = logger

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
        # Initialize parent class with the real messager
        super().__init__(messager=self.messager, client_id=self.client_id)
        self.answer_received_event = asyncio.Event()

        # Create user-specific answer topic
        self.user_answer_topic = f"{MESSAGING_TOPIC_ANSWER}/{self.user_id}"
        self.logger.info(f"CLI Client initialized with user_id: {self.user_id}")
        self.logger.info(f"User-specific answer topic: {self.user_answer_topic}")

    async def ask_question(self, question: str, topic: str) -> None:
        """Override parent method with consistent signature"""
        msg = AskQuestionMessage(
            question=question,
            user_id=self.user_id,
            source=self.client_id,
            data=None,
        )

        if self.messager:
            await self.messager.publish(topic, msg)
            self.logger.debug(
                f"Asking question: '{question}' on topic {topic} for user '{self.user_id}'"
            )
        else:
            self.logger.error("Messager not initialized. Cannot ask question.")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, setting shutdown flag...")
            if not self._shutdown_flag.is_set():
                self._shutdown_flag.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup_terminal(self):
        """Setup terminal for non-blocking input"""
        try:
            if sys.stdin.isatty():
                self._original_tty_settings = termios.tcgetattr(sys.stdin.fileno())
                tty.setraw(sys.stdin.fileno())
                # Ensure output works correctly in raw mode
                sys.stdout.flush()
        except (OSError, termios.error) as e:
            self.logger.debug(f"Could not setup terminal raw mode: {e}")
            self._original_tty_settings = None

    def _restore_terminal(self):
        """Restore terminal settings"""
        try:
            if self._original_tty_settings is not None:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, self._original_tty_settings
                )
                self._original_tty_settings = None
                # Ensure output is flushed after restoring
                sys.stdout.flush()
        except (OSError, termios.error) as e:
            self.logger.debug(f"Could not restore terminal settings: {e}")

    def _print_with_proper_formatting(self, text: str):
        """Print text with proper line formatting, temporarily restoring terminal if needed"""
        if self._original_tty_settings is not None:
            # Temporarily restore terminal for proper output
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, self._original_tty_settings
                )
                print(text)
                sys.stdout.flush()
                # Restore raw mode
                tty.setraw(sys.stdin.fileno())
            except (OSError, termios.error):
                # Fallback to regular print if terminal operations fail
                print(text)
                sys.stdout.flush()
        else:
            print(text)
            sys.stdout.flush()

    async def handle_answer(self, message: BaseMessage) -> None:
        """Handle answer messages with proper type"""
        self.logger.debug(f"Received message of type: {type(message)}")

        if isinstance(message, AnswerMessage):
            # No need for user_id filtering since we're subscribed to user-specific topic
            self.logger.debug(f"Processing answer message for user: {message.user_id}")

            # Use proper formatting for the answer display
            self._print_with_proper_formatting("\n--- Agent Final Answer ---")
            self._print_with_proper_formatting(f"Question: {message.question}")
            if message.answer:
                self._print_with_proper_formatting(f"Answer: {message.answer}")
            elif message.error:
                self._print_with_proper_formatting(f"Error: {message.error}")
            else:
                self._print_with_proper_formatting(
                    f"Received unexpected response format: {message.model_dump_json(indent=2)}"
                )
            self._print_with_proper_formatting("----------------------")

            if self.answer_received_event:
                self.answer_received_event.set()
        else:
            self.logger.warning(f"Received non-AnswerMessage: {type(message)}")

    async def handle_agent_llm_thought(self, message: BaseMessage) -> None:
        """Handle agent LLM response messages with proper type"""
        if isinstance(message, AgentLLMResponseMessage):
            self._print_with_proper_formatting("\n--- Agent Thinking (LLM Response) ---")
            if message.parsed_type == "tool_call" and message.parsed_tool_calls:
                self._print_with_proper_formatting("Agent plans to use tools:")
                for tc_item in message.parsed_tool_calls:
                    self._print_with_proper_formatting(
                        f"  - Tool ID: {tc_item.tool_id}, Arguments: {tc_item.arguments}"
                    )
            elif message.parsed_type == "direct" and message.parsed_direct_content:
                self._print_with_proper_formatting(
                    f"Agent direct thought/response: {message.parsed_direct_content}"
                )
            elif message.raw_content:
                self._print_with_proper_formatting(
                    f"Agent raw LLM output: {message.raw_content[:500]}..."
                )
            self._print_with_proper_formatting("-----------------------------------")

    async def handle_agent_tool_result(self, message: BaseMessage) -> None:
        self._print_with_proper_formatting("\n--- Agent Tool Execution Result ---")
        if isinstance(message, TaskResultMessage):
            self._print_with_proper_formatting(
                f"Tool '{message.tool_name}' (ID: {message.tool_id}) executed."
            )
            if message.result is not None:
                res_str = str(message.result)
                self._print_with_proper_formatting(
                    f"Result: {res_str[:500]}{'...' if len(res_str) > 500 else ''}"
                )
            else:
                self._print_with_proper_formatting("Result: (No content)")
        elif isinstance(message, TaskErrorMessage):
            self._print_with_proper_formatting(
                f"Tool '{message.tool_name}' (ID: {message.tool_id}) failed."
            )
            self._print_with_proper_formatting(f"Error: {message.error}")
        elif isinstance(message, BaseMessage):
            self.logger.warning(
                f"Received unhandled BaseMessage type on tool result topic: {type(message)}"
            )
            self._print_with_proper_formatting(
                f"Received unexpected tool result message type: {type(message)}"
            )
        self._print_with_proper_formatting("-----------------------------------")

    async def _read_user_input(self):
        """Non-blocking input reader that respects shutdown flag"""
        input_buffer = ""

        try:
            # Check if we're in a TTY environment or using piped input
            is_tty = sys.stdin.isatty()

            if is_tty:
                self._setup_terminal()

            while not self._shutdown_flag.is_set():
                try:
                    if is_tty and self._original_tty_settings is not None:
                        # Use raw terminal mode for better control in interactive mode
                        ready, _, _ = select.select([sys.stdin], [], [], 0.1)  # 100ms timeout
                        if ready:
                            char = sys.stdin.read(1)
                            if char:
                                # Handle special characters
                                if char == "\x03":  # Ctrl+C
                                    self.logger.info("Ctrl+C detected in input reader")
                                    self._shutdown_flag.set()
                                    break
                                elif char == "\x04":  # Ctrl+D (EOF)
                                    self.logger.info("Ctrl+D detected in input reader")
                                    self._shutdown_flag.set()
                                    break
                                elif char == "\r" or char == "\n":  # Enter
                                    if input_buffer.strip():
                                        user_input = input_buffer.strip()
                                        input_buffer = ""
                                        print()  # New line after input

                                        if user_input.lower() in ["quit", "exit"]:
                                            self._shutdown_flag.set()
                                            break

                                        if not self._shutdown_flag.is_set():
                                            await self._process_user_input(user_input)
                                    else:
                                        input_buffer = ""
                                        print()
                                        if not self._shutdown_flag.is_set():
                                            print("> ", end="", flush=True)
                                elif char == "\x7f" or char == "\b":  # Backspace
                                    if input_buffer:
                                        input_buffer = input_buffer[:-1]
                                        # Move cursor back, print space, move back again
                                        print("\b \b", end="", flush=True)
                                elif ord(char) >= 32:  # Printable characters
                                    input_buffer += char
                                    print(char, end="", flush=True)
                    else:
                        # Handle piped input or non-TTY environments
                        ready, _, _ = select.select([sys.stdin], [], [], 0.1)  # 100ms timeout
                        if ready:
                            try:
                                line = sys.stdin.readline()
                                if not line:  # EOF
                                    self.logger.info("EOF detected in input reader")
                                    self._shutdown_flag.set()
                                    break

                                user_input = line.strip()
                                if user_input:
                                    self.logger.info(f"Processing piped input: '{user_input}'")
                                    if user_input.lower() in ["quit", "exit"]:
                                        self._shutdown_flag.set()
                                        break

                                    if not self._shutdown_flag.is_set():
                                        await self._process_user_input(user_input)
                            except EOFError:
                                self.logger.info("EOFError detected in input reader")
                                self._shutdown_flag.set()
                                break

                    if self._shutdown_flag.is_set():
                        break

                except (OSError, select.error) as e:
                    # Handle errors gracefully - probably non-TTY environment
                    self.logger.debug(f"Input reading error (non-TTY?): {e}")
                    await asyncio.sleep(0.2)
                    if self._shutdown_flag.is_set():
                        break
                except Exception as e:
                    if not self._shutdown_flag.is_set():
                        self.logger.error(f"Error reading input: {e}")
                    break

        except asyncio.CancelledError:
            self.logger.info("Input reading task was cancelled")
            raise
        finally:
            if is_tty:
                self._restore_terminal()

    async def _process_user_input(self, user_input: str):
        """Process a user input question"""
        if self.answer_received_event:
            self.answer_received_event.clear()
        await self.ask_question(user_input, self.ask_topic)
        self.logger.debug(f"Waiting for final answer to: '{user_input}'...")

        try:
            if self.answer_received_event:
                await asyncio.wait_for(
                    self.answer_received_event.wait(),
                    timeout=120.0,
                )
        except asyncio.TimeoutError:
            self._print_with_proper_formatting(
                "\n--- No final answer received within timeout. Check for thinking steps. ---"
            )
        finally:
            if not self._shutdown_flag.is_set() and sys.stdin.isatty():
                print("> ", end="", flush=True)

    async def _cleanup(self):
        """Centralized cleanup method"""
        if self._cleanup_started:
            return
        self._cleanup_started = True

        self.logger.info("Starting cleanup...")

        # Cancel input task if running
        if self._input_task and not self._input_task.done():
            self.logger.info("Cancelling input task...")
            self._input_task.cancel()
            try:
                await asyncio.wait_for(self._input_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.logger.info("Input task cancelled or timed out")

        # Restore terminal
        self._restore_terminal()

        # Disconnect messager
        if self.messager and self.messager.is_connected():
            self.logger.info("Disconnecting messager...")
            try:
                await asyncio.wait_for(self.messager.disconnect(), timeout=5.0)
                self.logger.info("Messager disconnected successfully")
            except asyncio.TimeoutError:
                self.logger.warning("Messager disconnect timed out")
            except Exception as e:
                self.logger.error(f"Error disconnecting messager: {e}")

        self.logger.info("Cleanup complete")

    async def run_interactive(self) -> bool:
        """Main interactive loop with improved shutdown handling"""
        success = False
        try:
            await self.initialize()
            if not self.messager:
                self.logger.error("CLI Error: Messager failed to initialize.")
                return False

            if not await self.messager.connect():
                self.logger.error("CLI Error: Failed to connect. Check logs for details.")
                return False

            # Subscribe to user-specific answer topic
            if not self.user_answer_topic:
                self.logger.error("CLI Error: User answer topic not initialized.")
                return False

            await self.messager.subscribe(
                self.user_answer_topic, self.handle_answer, message_cls=AnswerMessage
            )
            self.logger.info(f"Subscribed to user-specific answer topic: {self.user_answer_topic}")

            if MESSAGING_TOPIC_AGENT_LLM_RESPONSE:
                # For LLM responses, we might want user-specific topics too, but keeping global for now
                await self.messager.subscribe(
                    MESSAGING_TOPIC_AGENT_LLM_RESPONSE,
                    self.handle_agent_llm_thought,
                    message_cls=AgentLLMResponseMessage,
                )
                self.logger.info(
                    f"Subscribed to agent LLM response topic: {MESSAGING_TOPIC_AGENT_LLM_RESPONSE}"
                )

            if MESSAGING_TOPIC_AGENT_TOOL_RESULT:
                # For tool results, we might want user-specific topics too, but keeping global for now
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
            print("Type 'quit', 'exit', or press Ctrl+C to leave.")
            print("> ", end="", flush=True)

            # Start the input reading task
            self._input_task = asyncio.create_task(self._read_user_input())

            # Wait for shutdown signal or input task completion
            try:
                await asyncio.wait(
                    [asyncio.create_task(self._shutdown_flag.wait()), self._input_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                self.logger.info("Interactive loop was cancelled")
                raise

            success = True

        except asyncio.CancelledError:
            self.logger.info("run_interactive task was cancelled.")
            success = False
            raise
        except Exception as e:
            self.logger.error(
                f"CLI Error: An unexpected error occurred in run_interactive: {e}", exc_info=True
            )
            success = False
        finally:
            await self._cleanup()

        return success

    def start(self) -> bool:
        """Synchronous entry point for the CLI - expected by main module"""
        # Setup signal handlers
        self._setup_signal_handlers()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = False

        try:
            success = loop.run_until_complete(self.run_interactive())
            # Ensure we return a boolean
            success = success if success is not None else False
        except KeyboardInterrupt:
            self.logger.info("CLI: KeyboardInterrupt caught in start.")
            if not self._shutdown_flag.is_set():
                self._shutdown_flag.set()
            success = False
        except Exception as e:
            self.logger.error(f"CLI Error: An unexpected error in start: {e}", exc_info=True)
            success = False
        finally:
            self.logger.info("CLI: Cleaning up event loop...")

            # Clean shutdown of remaining tasks
            try:
                if not loop.is_closed():
                    # Cancel all remaining tasks
                    pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]

                    if pending_tasks:
                        self.logger.info(f"Cancelling {len(pending_tasks)} pending tasks...")
                        for task in pending_tasks:
                            task.cancel()

                        # Wait for tasks to complete cancellation with timeout
                        try:
                            loop.run_until_complete(
                                asyncio.wait_for(
                                    asyncio.gather(*pending_tasks, return_exceptions=True),
                                    timeout=3.0,
                                )
                            )
                        except asyncio.TimeoutError:
                            self.logger.warning("Some tasks did not cancel within timeout")

                    # Shutdown async generators
                    loop.run_until_complete(loop.shutdown_asyncgens())

                    # Close the loop
                    loop.close()
                    self.logger.info("Event loop closed successfully")

            except Exception as e:
                self.logger.error(f"Error during loop cleanup: {e}")

        return success


if __name__ == "__main__":
    cli_client = CliClient()
    exit_code = 0 if cli_client.start() else 1
    sys.exit(exit_code)
