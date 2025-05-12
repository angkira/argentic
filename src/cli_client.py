import asyncio
import sys
from typing import Optional
import yaml
import uuid
import re
import concurrent.futures

from core.client import Client
from core.messager.messager import Messager
from core.protocol.message import AnswerMessage, AskQuestionMessage
from core.logger import LogLevel, get_logger

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

# --- MESSAGING Configuration ---
messaging_config = config["messaging"]
MESSAGING_BROKER = messaging_config["broker_address"]
MESSAGING_PORT = messaging_config["port"]
MESSAGING_KEEPALIVE = messaging_config["keepalive"]
MESSAGING_CLIENT_ID = messaging_config.get("cli_client_id", f"cli_client_{uuid.uuid4()}")

# Get topics from config
MESSAGING_TOPIC_ASK = config["topics"]["commands"].get("ask_question")
MESSAGING_TOPIC_ANSWER = config["topics"]["responses"]["answer"]
MESSAGING_PUB_LOG = config["topics"]["log"]

logger = get_logger("CliClient", LogLevel.INFO)


class CliClient(Client):
    """CLI Client implementation inheriting from base Client class"""

    def __init__(self):
        """Initialize the CLI Client"""
        self.client_id = MESSAGING_CLIENT_ID
        self.ask_topic = MESSAGING_TOPIC_ASK
        self.messager: Optional[Messager] = None
        self.answer_received_event: Optional[asyncio.Event] = None

    async def initialize(self):
        """Initialize the client in async context"""
        self.messager = Messager(
            protocol=messaging_config["protocol"],
            broker_address=MESSAGING_BROKER,
            port=MESSAGING_PORT,
            client_id=self.client_id,
            keepalive=MESSAGING_KEEPALIVE,
            pub_log_topic=MESSAGING_PUB_LOG,
            log_level=LogLevel.INFO,
        )
        super().__init__(
            messager=self.messager,
            client_id=self.client_id,
        )
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logger

        self.answer_received_event = asyncio.Event()

    async def ask_question(self, question_text: str, topic: str, user_id: Optional[str] = None):
        """Publishes a question to the specified topic."""
        msg = AskQuestionMessage(
            question=question_text, user_id=user_id or self.client_id, source=self.client_id
        )
        if self.messager:
            await self.messager.publish(topic, msg.model_dump_json())
            self.logger.info(f"Asking question: '{question_text}' on topic {topic}")
        else:
            self.logger.error("Messager not initialized. Cannot ask question.")

    async def handle_answer(self, message: AnswerMessage) -> None:
        """Override to handle CLI-formatted responses including LangChain format"""
        print("\n--- Agent Response ---")
        print(f"Question: {message.question}")

        if message.answer:
            answer = message.answer
            if "content='" in answer:
                try:
                    content_match = re.search(r"content='([^']*)'", answer)
                    if content_match:
                        answer = content_match.group(1)
                except Exception:
                    pass

            if "```" in answer:
                code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
                code_blocks = re.findall(code_block_pattern, answer)
                if code_blocks:
                    answer = "\n".join(code_blocks)
                else:
                    answer = answer.replace("```json", "").replace("```", "").strip()
            print(f"Answer: {answer}")
        elif message.error:
            print(f"Error: {message.error}")
        else:
            print(f"Received unexpected response format: {message.model_dump_json(indent=2)}")

        print("----------------------")
        if self.answer_received_event:
            self.answer_received_event.set()

    async def run_interactive(self):
        """Run the interactive CLI client"""
        loop = asyncio.get_running_loop()

        # Create a ThreadPoolExecutor.
        # In Python 3.8+, worker threads inherit the daemon status of the creating thread if it's a daemon.
        # If the creating thread is NOT a daemon (e.g., main thread), worker threads ARE daemonic by default.
        # This means we likely don't need a custom initializer for daemon status with modern Python.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, # Only one thread needed for stdin
            thread_name_prefix='CLIInputWorker' # Optional: for easier debugging
        )

        try:
            await self.initialize()

            if not await self.connect():
                self.logger.error("CLI Error: Failed to connect. Check logs for details.")
                return False

            await self.messager.subscribe(
                MESSAGING_TOPIC_ANSWER, self.handle_answer, message_cls=AnswerMessage
            )
            self.logger.info(f"Subscribed to answer topic: {MESSAGING_TOPIC_ANSWER}")

            print("--- Agent CLI Client ---")
            print("Type your question and press Enter.")
            print("Type 'quit' or 'exit' to leave.")

            while True:
                # Run sys.stdin.readline in the daemon thread via our custom executor
                user_input = await loop.run_in_executor(executor, sys.stdin.readline)
                user_input = user_input.strip()

                if user_input.lower() in ["quit", "exit"]:
                    break

                if user_input:
                    if self.answer_received_event:
                        self.answer_received_event.clear()

                    await self.ask_question(user_input, self.ask_topic)
                    self.logger.info(f"Waiting for answer to: '{user_input}'...")

                    try:
                        if self.answer_received_event:
                            await asyncio.wait_for(self.answer_received_event.wait(), timeout=60.0)
                    except asyncio.TimeoutError:
                        print("\n--- No response received within timeout. ---")
                    finally:
                        print("> ", end="", flush=True)
                else:
                    print("> ", end="", flush=True)
            return True

        except asyncio.CancelledError:
            self.logger.info("CLI: run_interactive task was cancelled.")
            # Propagate cancellation if needed, or handle specific cleanup
            raise # Re-raise CancelledError so run_until_complete in start() sees it if not handled by KeyboardInterrupt
        except Exception as e:
            self.logger.error(f"CLI Error: An unexpected error occurred in run_interactive: {e}", exc_info=True)
            return False
        finally:
            self.logger.info("CLI: Shutting down (run_interactive's finally block)...")
            await self.stop() # Graceful MESSAGING disconnect etc.
            self.logger.info("CLI: Shutdown complete (run_interactive's finally block).")
            
            self.logger.info("CLI: Shutting down stdin thread pool executor...")
            # For daemon threads, wait=False is appropriate as they won't block exit.
            # cancel_futures (Python 3.9+) attempts to cancel pending work.
            if sys.version_info >= (3, 9):
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=False) # For Python < 3.9
            self.logger.info("CLI: Stdin thread pool executor shutdown initiated.")

    def start(self):
        """Synchronous entry point"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        keyboard_interrupted = False
        try:
            # loop.run_until_complete will propagate KeyboardInterrupt
            # after run_interactive (if it's the current task) handles CancelledError
            # and runs its finally block.
            return loop.run_until_complete(self.run_interactive())
        except KeyboardInterrupt:
            self.logger.info("CLI: KeyboardInterrupt caught in start. Expecting run_interactive's finally block to have handled self.stop().")
            keyboard_interrupted = True
            # At this point, self.run_interactive() should have been cancelled
            # and its 'finally' block (containing self.stop()) should have executed.
            # If self.stop() is lengthy or hangs, the program will wait here until it's done,
            # or until this very KeyboardInterrupt exception fully unwinds run_until_complete.
        except Exception as e:
            self.logger.error(f"CLI Error: An unexpected error in start: {e}", exc_info=True)
            # For other exceptions, we don't forcibly exit immediately, just return False after cleanup.
            return False
        finally:
            self.logger.info("CLI: Cleaning up event loop in start's finally block.")
            if not loop.is_closed():
                try:
                    # Attempt to cancel all remaining tasks before closing the loop
                    all_tasks = asyncio.all_tasks(loop=loop)
                    # Filter out the current task if this finally block itself is run as part of a task (should not be the case here)
                    # tasks_to_cancel = [t for t in all_tasks if t is not asyncio.current_task(loop=loop)] # More robust
                    tasks_to_cancel = [t for t in all_tasks]


                    if tasks_to_cancel:
                        self.logger.info(f"CLI: Cancelling {len(tasks_to_cancel)} outstanding tasks during final cleanup...")
                        for task in tasks_to_cancel:
                            task.cancel()
                        # Give cancelled tasks a chance to run their cleanup code
                        loop.run_until_complete(asyncio.gather(*tasks_to_cancel, return_exceptions=True))
                        self.logger.info("CLI: Outstanding tasks gathered during final cleanup.")

                    # Shutdown async generators
                    if loop.is_running(): # Check if running before shutting down async gens
                         self.logger.info("CLI: Shutting down async generators...")
                         loop.run_until_complete(loop.shutdown_asyncgens())
                         self.logger.info("CLI: Async generators shutdown.")

                except RuntimeError as e_rt:
                    self.logger.warning(f"CLI: Runtime error during final loop cleanup (loop may be closed or stopping): {e_rt}")
                except Exception as e_final_cleanup:
                    self.logger.error(f"CLI: Error during final cleanup of loop/tasks: {e_final_cleanup}", exc_info=True)
                finally:
                    if not loop.is_closed(): # Re-check as operations above might have closed it
                        self.logger.info("CLI: Closing event loop.")
                        loop.close()
                        self.logger.info("CLI: Event loop closed.")
                    else:
                        self.logger.info("CLI: Event loop was already closed by prior cleanup in finally.")
            else:
                 self.logger.info("CLI: Event loop was already closed before start's finally block.")

        if keyboard_interrupted:
            self.logger.info("CLI: Exiting program explicitly after KeyboardInterrupt handling.")
            sys.exit(0) # Force exit after cleanup attempts

        # This return is for cases where run_interactive completed without KeyboardInterrupt (e.g., user typed 'exit')
        # or if another non-KeyboardInterrupt exception occurred and was handled.
        return False


# --- Main Execution Block ---
if __name__ == "__main__":
    cli_client = CliClient()
    cli_client.start()
