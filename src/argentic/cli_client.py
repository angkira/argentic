import asyncio
import select
import signal
import sys
import termios
import tty
import uuid
from collections import deque
from typing import Any, Callable, Dict, Optional, cast

import yaml

from argentic.core.client import Client
from argentic.core.logger import LogLevel, get_logger
from argentic.core.messager.messager import Messager
from argentic.core.messager.protocols import MessagerProtocol
from argentic.core.protocol.message import (
    AgentLLMResponseMessage,
    AnswerMessage,
    AskQuestionMessage,
    BaseMessage,
)
from argentic.core.protocol.task import TaskErrorMessage, TaskResultMessage


class CliClient(Client):
    def __init__(self, config_path: str = "config.yaml", log_level: str = "INFO"):
        self.config_path = config_path
        self.log_level_str = log_level
        self.config = self._load_config(config_path)

        log_level_enum = LogLevel[log_level.upper()]
        self.logger = get_logger("CliClient", log_level_enum)

        # --- MESSAGING Configuration with robust access ---
        messaging_config = self._get_config_value(self.config, "messaging", {}, required=True)
        self.messaging_broker: str = self._get_config_value(
            messaging_config, "broker_address", required=True
        )
        self.messaging_port: int = self._get_config_value(
            messaging_config, "port", 1883, required=False
        )
        self.messaging_keepalive: int = self._get_config_value(
            messaging_config, "keepalive", 60, required=False
        )
        messaging_client_id: str = (
            f"{messaging_config.get('cli_client_id', 'cli_client')}_{uuid.uuid4()}"
        )

        self.ask_topic: str = self._get_config_value(
            self.config, "topics.commands.ask_question", required=True
        )
        self.messaging_topic_answer: str = self._get_config_value(
            self.config, "topics.responses.answer", required=True
        )
        self.messaging_pub_log: Optional[str] = self._get_config_value(
            self.config, "topics.log", required=False
        )

        self.messaging_topic_agent_llm_response: Optional[str] = self._get_config_value(
            self.config, "topics.agent_events.llm_response", required=False
        )
        self.messaging_topic_agent_tool_result: Optional[str] = self._get_config_value(
            self.config, "topics.agent_events.tool_result", required=False
        )

        # Initialize a dummy Messager for the parent class init, it will be replaced in initialize()
        # This is a workaround for the base Client expecting a Messager instance directly in __init__
        # while CliClient needs async initialization for the real Messager.
        dummy_messager = Messager(
            broker_address="localhost",  # Placeholder
            port=1883,  # Placeholder
            client_id=f"dummy_client_{uuid.uuid4()}",  # Unique dummy client ID
            log_level=LogLevel.CRITICAL,  # Suppress logs from dummy
        )
        super().__init__(messager=dummy_messager, client_id=messaging_client_id)

        self._shutdown_flag = asyncio.Event()
        self._input_task: Optional[asyncio.Task] = None
        self._waiting_task: Optional[asyncio.Task] = None
        self._spinner: Optional[Any] = None
        self._answer_timeout_seconds: float = 120.0
        self._cleanup_started = False
        self.answer_received_event: Optional[asyncio.Event] = None
        self._original_tty_settings: Optional[Any] = None
        self.user_answer_topic: str = ""  # Will be set in initialize()

        # Will be properly initialized in initialize()
        self.messager: Optional[Messager] = None
        # De-duplicate received messages by ID (LRU window)
        self._recent_message_ids: "deque[str]" = deque(maxlen=1024)

    # -------------------------
    # Spinner and waiting logic
    # -------------------------
    def _is_waiting(self) -> bool:
        return bool(self._waiting_task and not self._waiting_task.done())

    def _start_wait_for_answer(self, timeout_seconds: float) -> None:
        if self.answer_received_event is None:
            self.answer_received_event = asyncio.Event()
        # Ensure no double waiting
        if self._is_waiting():
            return
        self._waiting_task = asyncio.create_task(self._wait_for_answer(timeout_seconds))

    async def _wait_for_answer(self, timeout_seconds: float) -> None:
        try:
            # Wait for either answer, shutdown, or timeout
            tasks = []
            if self.answer_received_event is not None:
                tasks.append(asyncio.create_task(self.answer_received_event.wait()))
            tasks.append(asyncio.create_task(self._shutdown_flag.wait()))

            done: set
            pending: set
            try:
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=timeout_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                # Cancel any waiting tasks left
                for t in tasks:
                    if not t.done():
                        t.cancel()

            if self._shutdown_flag.is_set():
                return

            # If answer not received in time
            if self.answer_received_event and not self.answer_received_event.is_set():
                await self._stop_spinner()
                self._print_with_proper_formatting(
                    "\n--- No final answer received within timeout. Check for thinking steps. ---"
                )
                if not self._shutdown_flag.is_set() and sys.stdin.isatty():
                    print("> ", end="", flush=True)
        except asyncio.CancelledError:
            # Expected on cancellation
            raise

    async def _cancel_waiting(self) -> None:
        # Cancel waiter and spinner
        if self._waiting_task and not self._waiting_task.done():
            self._waiting_task.cancel()
            try:
                await asyncio.wait_for(self._waiting_task, timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        await self._stop_spinner()
        self._print_with_proper_formatting("\nCancelled waiting.")

    def _start_spinner(self, prefix_text: str) -> None:
        # Avoid double start
        if self._spinner is not None:
            return
        # In non-TTY, print once and skip animation
        if not sys.stdin.isatty():
            self._print_with_proper_formatting(f"{prefix_text}")
            return
        try:
            from yaspin import yaspin  # type: ignore
        except Exception:
            # Fallback: simple print if yaspin not available
            self._print_with_proper_formatting(f"{prefix_text}")
            return

        try:
            # Use a built-in fast spinner preset for responsiveness
            try:
                from yaspin.spinners import Spinners  # type: ignore

                spinner = yaspin(spinner=Spinners.dots, text=prefix_text, color="yellow")
            except Exception:
                spinner = yaspin(text=prefix_text, color="yellow")
            spinner.start()
            self._spinner = spinner
        except Exception:
            # Fallback: simple print
            self._spinner = None
            self._print_with_proper_formatting(f"{prefix_text}")

    async def _stop_spinner(self, ok: bool | None = None, text: Optional[str] = None) -> None:
        try:
            if self._spinner is not None:
                # Persist result if requested, else just stop quietly
                if ok is True and text is not None:
                    try:
                        self._spinner.ok(text)
                    except Exception:
                        self._spinner.stop()
                elif ok is False and text is not None:
                    try:
                        self._spinner.fail(text)
                    except Exception:
                        self._spinner.stop()
                else:
                    self._spinner.stop()
        finally:
            self._spinner = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r") as f:
                loaded_config = yaml.safe_load(f)
                if isinstance(loaded_config, dict):
                    print(f"CLI: Configuration loaded from '{config_path}'.")
                    return loaded_config
                else:
                    raise ValueError(
                        f"CLI Error: Configuration in '{config_path}' is not a valid dictionary."
                    )
        except FileNotFoundError:
            raise FileNotFoundError(f"CLI Error: Configuration file '{config_path}' not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"CLI Error: Parsing configuration file '{config_path}': {e}")

    def _get_config_value(
        self, cfg_dict: dict, path: str, default: Any = None, required: bool = True
    ) -> Any:
        keys = path.split(".")
        val = cfg_dict
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                if required:
                    raise ValueError(
                        f"CLI Error: Missing required config: '{path}' in '{self.config_path}'."
                    )
                return default
        return val

    async def initialize(self):
        """Initialize the client in async context"""
        # Retrieve protocol as string and convert to enum
        protocol_str: str = self._get_config_value(
            self.config["messaging"], "protocol", "mqtt", required=False
        )

        # Convert string to MessagerProtocol enum with proper type handling
        if protocol_str == "mqtt":
            protocol_enum = MessagerProtocol.MQTT
        elif protocol_str == "kafka":
            protocol_enum = MessagerProtocol.KAFKA
        elif protocol_str == "redis":
            protocol_enum = MessagerProtocol.REDIS
        elif protocol_str == "rabbitmq":
            protocol_enum = MessagerProtocol.RABBITMQ
        else:
            protocol_enum = MessagerProtocol.MQTT  # Default fallback

        # Create the real Messager instance
        real_messager = Messager(
            protocol=protocol_enum,  # Use the properly typed enum
            broker_address=self.messaging_broker,
            port=self.messaging_port,
            client_id=self.client_id,
            keepalive=self.messaging_keepalive,
            pub_log_topic=self.messaging_pub_log,
            log_level=self.log_level_str,
        )
        self.messager = real_messager  # Assign the real messager

        self.answer_received_event = asyncio.Event()

        # Create user-specific answer topic
        self.user_answer_topic = f"{self.messaging_topic_answer}/{self.user_id}"
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

        # Fallback synchronous signal handlers (used only in _start_sync context)
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, setting shutdown flag...")
            if not self._shutdown_flag.is_set():
                try:
                    loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(cast(Callable[..., object], self._shutdown_flag.set))
                except RuntimeError:
                    # No running loop; set directly as a last resort
                    self._shutdown_flag.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _install_asyncio_signal_handlers(self, loop: asyncio.AbstractEventLoop):
        """Install asyncio-aware signal handlers to trigger shutdown cleanly."""

        def _set_shutdown() -> None:
            if not self._shutdown_flag.is_set():
                self._shutdown_flag.set()

        try:
            cb_any = cast(Callable[..., object], _set_shutdown)
            loop.add_signal_handler(signal.SIGINT, cb_any)
            loop.add_signal_handler(signal.SIGTERM, cb_any)
        except NotImplementedError:
            # Fallback for platforms without add_signal_handler
            self._setup_signal_handlers()

    def _setup_terminal(self):
        """Setup terminal for non-blocking input"""
        self.logger.debug(
            f"Attempting to set up terminal. sys.stdin.isatty(): {sys.stdin.isatty()}"
        )
        try:
            if sys.stdin.isatty():
                self._original_tty_settings = termios.tcgetattr(sys.stdin.fileno())
                tty.setraw(sys.stdin.fileno())
                # Ensure output works correctly in raw mode
                sys.stdout.flush()
                self.logger.debug("Terminal raw mode enabled.")
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
        # If spinner is active, write via spinner to avoid breaking the animation line
        if self._spinner is not None and sys.stdin.isatty():
            try:
                self._spinner.write(text)
                return
            except Exception:
                # Fall back to normal printing path
                pass
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
            # De-duplicate on message_id
            try:
                msg_id = getattr(message, "message_id", None)
                if msg_id is not None:
                    if msg_id in self._recent_message_ids:
                        self.logger.debug(f"Duplicate answer message ignored: {msg_id}")
                        return
                    self._recent_message_ids.append(msg_id)
            except Exception:
                pass
            # Filter out answers for other users (when subscribed to base topic too)
            if message.user_id and message.user_id != self.user_id:
                self.logger.debug(
                    f"Ignoring answer for different user: {message.user_id} != {self.user_id}"
                )
                return
            # No need for user_id filtering since we're subscribed to user-specific topic
            self.logger.debug(f"Processing answer message for user: {message.user_id}")
            # Stop any spinner before printing the answer
            try:
                await self._stop_spinner()
            except Exception:
                pass
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
            # Signal any waiter tasks that the answer has arrived
            if self.answer_received_event:
                self.answer_received_event.set()
            # Re-prompt if interactive and not shutting down
            if not self._shutdown_flag.is_set() and sys.stdin.isatty():
                print("> ", end="", flush=True)
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
            self.logger.debug(f"_read_user_input started. is_tty: {is_tty}")

            if is_tty:
                self._setup_terminal()

            while not self._shutdown_flag.is_set():
                try:
                    if is_tty and self._original_tty_settings is not None:
                        # Use raw terminal mode for better control in interactive mode
                        ready, _, _ = select.select([sys.stdin], [], [], 0.01)  # 10ms timeout
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

                                        # Allow cancelling current waiting state
                                        if user_input.lower() == "cancel":
                                            await self._cancel_waiting()
                                            if not self._shutdown_flag.is_set():
                                                print("> ", end="", flush=True)
                                            continue

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
                        ready, _, _ = select.select([sys.stdin], [], [], 0.05)  # 50ms timeout
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

                                    if user_input.lower() == "cancel":
                                        await self._cancel_waiting()
                                        continue

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
                    await asyncio.sleep(0.05)
                    if self._shutdown_flag.is_set():
                        break
                except Exception as e:
                    if not self._shutdown_flag.is_set():
                        self.logger.error(f"Error reading input: {e}")
                    break

                # Yield control to let other tasks (MQTT handlers, spinner) run smoothly
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            self.logger.info("Input reading task was cancelled")
            raise
        finally:
            if is_tty:
                self._restore_terminal()

    async def _process_user_input(self, user_input: str):
        """Process a user input question non-blockingly: publish and start waiter/spinner."""
        # If a previous question is still waiting, ask user to cancel first
        if self._is_waiting():
            self._print_with_proper_formatting(
                "Already waiting for a response. Type 'cancel' to abort or wait."
            )
            return

        if self.answer_received_event:
            self.answer_received_event.clear()
        await self.ask_question(user_input, self.ask_topic)
        self.logger.debug(f"Waiting for final answer to: '{user_input}'...")

        # Start spinner and waiter in background
        self._start_spinner("Waiting for response...")
        self._start_wait_for_answer(self._answer_timeout_seconds)

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

        # Cancel waiter and spinner if running
        if self._waiting_task and not self._waiting_task.done():
            self.logger.info("Cancelling waiting task...")
            self._waiting_task.cancel()
            try:
                await asyncio.wait_for(self._waiting_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.logger.info("Waiting task cancelled or timed out")
        # Stop spinner if present
        if self._spinner is not None:
            try:
                await self._stop_spinner()
            except Exception:
                self._spinner = None

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
            self.logger.debug("Starting run_interactive")
            await self.initialize()
            self.logger.debug("Initialization complete")

            # Install asyncio-aware signal handlers so Ctrl+C always triggers shutdown
            try:
                loop = asyncio.get_running_loop()
                self._install_asyncio_signal_handlers(loop)
            except Exception as e:
                self.logger.debug(f"Could not install asyncio signal handlers: {e}")

            if not self.messager:
                self.logger.error("CLI Error: Messager failed to initialize.")
                return False

            self.logger.debug("About to connect to messager")
            if not await self.messager.connect():
                self.logger.error("CLI Error: Failed to connect. Check logs for details.")
                return False
            self.logger.debug("Connected to messager successfully")

            # Subscribe to user-specific answer topic
            if not self.user_answer_topic:
                self.logger.error("CLI Error: User answer topic not initialized.")
                return False

            self.logger.debug(f"About to subscribe to user answer topic: {self.user_answer_topic}")
            await self.messager.subscribe(
                self.user_answer_topic, self.handle_answer, message_cls=AnswerMessage
            )
            self.logger.info(f"Subscribed to user-specific answer topic: {self.user_answer_topic}")

            # Subscribe to the base answer topic as a fallback (filter by user_id in handler)
            try:
                await self.messager.subscribe(
                    self.messaging_topic_answer, self.handle_answer, message_cls=AnswerMessage
                )
                self.logger.info(
                    f"Subscribed to base answer topic (fallback): {self.messaging_topic_answer}"
                )
            except Exception as e:
                self.logger.debug(f"Fallback base answer topic subscribe skipped/failed: {e}")

            if self.messaging_topic_agent_llm_response:
                self.logger.debug(
                    "About to subscribe to LLM response topic: "
                    f"{self.messaging_topic_agent_llm_response}"
                )
                # For LLM responses, we might want user-specific topics too, but keeping global for now
                await self.messager.subscribe(
                    self.messaging_topic_agent_llm_response,
                    self.handle_agent_llm_thought,
                    message_cls=AgentLLMResponseMessage,
                )
                self.logger.info(
                    "Subscribed to agent LLM response topic: "
                    f"{self.messaging_topic_agent_llm_response}"
                )

            if self.messaging_topic_agent_tool_result:
                self.logger.debug(
                    "About to subscribe to tool result topic: "
                    f"{self.messaging_topic_agent_tool_result}"
                )
                # For tool results, we might want user-specific topics too, but keeping global for now
                await self.messager.subscribe(
                    self.messaging_topic_agent_tool_result,
                    self.handle_agent_tool_result,
                    message_cls=BaseMessage,
                )
                self.logger.info(
                    "Subscribed to agent tool result topic: "
                    f"{self.messaging_topic_agent_tool_result}"
                )

            self.logger.debug("All subscriptions complete, about to show interactive prompt")
            print("--- Agent CLI Client ---")
            print("Type your question and press Enter.")
            print("Type 'quit', 'exit', or press Ctrl+C to leave.")
            print("> ", end="", flush=True)
            self.logger.debug("Interactive prompt shown")

            # Start the input reading task
            self.logger.debug("About to start _read_user_input task")
            self._input_task = asyncio.create_task(self._read_user_input())
            self.logger.debug(
                "Started _read_user_input task, now waiting for shutdown or completion"
            )

            # Wait for shutdown signal or input task completion
            try:
                await asyncio.wait(
                    [asyncio.create_task(self._shutdown_flag.wait()), self._input_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                self.logger.debug("Finished waiting for shutdown or input task completion")
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

    async def start(self) -> bool:
        """Asynchronous entry point for the CLI - overrides parent Client.start"""
        return await self.run_interactive()

    def _start_sync(self) -> bool:
        """Synchronous entry point for the CLI - expected by main module"""
        # Setup basic (fallback) signal handlers. Async-aware handlers are installed later.
        self._setup_signal_handlers()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = False

        try:
            success = loop.run_until_complete(self.start())  # Call the async start method
            # Ensure we return a boolean
            success = success if success is not None else False
        except KeyboardInterrupt:
            self.logger.info("CLI: KeyboardInterrupt caught in _start_sync.")
            if not self._shutdown_flag.is_set():
                self._shutdown_flag.set()
            success = False
        except Exception as e:
            self.logger.error(f"CLI Error: An unexpected error in _start_sync: {e}", exc_info=True)
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
    try:
        cli_client = CliClient()
        exit_code = 0 if cli_client._start_sync() else 1  # Call the synchronous start method
        sys.exit(exit_code)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
