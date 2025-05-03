import json
from functools import wraps
from typing import Callable, Dict, Any, Optional

# Local application imports
from core.messager import Messager, MQTTMessage

# Update imports to use the new classes
from core.rag import RAGManager
from core.agent import Agent


# Modified decorator factory to support both RAGManager and Agent
def mqtt_handler_decorator(
    messager: Messager,
    rag_manager: Optional[RAGManager] = None,
    agent: Optional[Agent] = None,
    **handler_kwargs,
) -> Callable:
    """
    Decorator factory for MQTT message handlers.

    Handles common boilerplate and injects dependencies.
    - Decodes payload (UTF-8)
    - Parses JSON payload
    - Basic error handling for decoding/parsing.
    - Injects Messager, RAGManager (optional), Agent (optional), parsed data, original message,
      and any extra keyword arguments passed to the decorator factory
      into the decorated handler function.
    """

    def decorator(
        # The decorated function now expects injected args + handler_kwargs
        func: Callable[
            [
                Messager,
                Optional[Dict[str, Any]],  # Parsed data
                MQTTMessage,  # Original message
                Dict[str, Any],  # handler_kwargs (handler specific kwargs)
            ],
            None,
        ],
    ) -> Callable[[MQTTMessage], None]:
        @wraps(func)
        def wrapper(msg: MQTTMessage) -> None:
            topic = msg.topic
            payload_str: Optional[str] = None
            data: Optional[Dict[str, Any]] = None

            try:
                # 1. Decode Payload
                payload_str = msg.payload.decode("utf-8")
                # Basic log kept for visibility, detailed logging via injected messager
                print(f"MQTT Received on {topic}: {payload_str}")

                # 2. Parse JSON
                try:
                    data = json.loads(payload_str)
                except json.JSONDecodeError:
                    err_msg = f"Invalid JSON received on {topic}: {payload_str}"
                    print(err_msg)
                    # Use the injected messager for logging
                    messager.log(err_msg, level="warning")
                    data = None  # Pass None for data if JSON parsing fails

                # 3. Call original handler with injected dependencies and kwargs
                import inspect

                sig = inspect.signature(func)
                params = sig.parameters
                call_kwargs = {}

                # Prepare kwargs based on available parameters in the function signature
                if "messager" in params:
                    call_kwargs["messager"] = messager
                if "data" in params:
                    call_kwargs["data"] = data
                if "msg" in params:
                    call_kwargs["msg"] = msg

                # Conditionally pass handler_kwargs ONLY if the function expects it
                if "handler_kwargs" in params:
                    call_kwargs["handler_kwargs"] = handler_kwargs

                # Add optional dependencies if they exist and are in the signature
                if "rag_manager" in params and rag_manager is not None:
                    call_kwargs["rag_manager"] = rag_manager
                if "agent" in params and agent is not None:
                    call_kwargs["agent"] = agent

                # --- DEBUGGING ---
                print(f"DEBUG: Decorator wrapping func: {func.__name__}")
                print(f"DEBUG: Inspected params: {list(params.keys())}")
                print(f"DEBUG: Call kwargs being passed: {list(call_kwargs.keys())}")
                # --- END DEBUGGING ---

                # Call the function with the collected kwargs
                try:
                    func(**call_kwargs)
                except Exception as e:
                    print(
                        f"DEBUG: Error during func call: {e}"
                    )  # Add debug for the specific call error
                    raise  # Re-raise the original exception

            except UnicodeDecodeError:
                err_msg = f"Cannot decode payload on {topic} as UTF-8: {msg.payload!r}"
                print(err_msg)
                messager.log(err_msg, level="error")
            except Exception as e:
                # Catch-all for unexpected errors within the handler logic itself
                err_msg = (
                    f"Unhandled error processing message on {topic}: {e.__class__.__name__}: {e}"
                )
                print(err_msg)
                messager.log(err_msg, level="error")
                import traceback

                traceback.print_exc()  # Print stack trace for debugging

        return wrapper

    return decorator
