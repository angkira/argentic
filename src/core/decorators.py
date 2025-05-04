import json
from functools import wraps
from typing import Callable, Dict, Any, Optional, Type
import inspect

from pydantic import ValidationError

# Local application imports
from core.messager import Messager, MQTTMessage
from core.agent import Agent
from core.protocol.message import from_mqtt_message, AnyMessage, BaseMessage


# Modified decorator factory to inject parsed Pydantic message object
def mqtt_handler_decorator(
    messager: Messager,
    agent: Optional[Agent] = None,
    **handler_kwargs,
) -> Callable:
    """
    Decorator factory for MQTT message handlers using Pydantic.

    Handles common boilerplate and injects dependencies.
    - Parses the incoming MQTT message using `from_mqtt_message`.
    - Handles parsing/validation errors.
    - Injects Messager, the parsed Pydantic message object, Agent (optional),
      original MQTT message, and any extra keyword arguments passed to the
      decorator factory into the decorated handler function.
    """

    def decorator(
        # The decorated function now expects injected args, including the parsed message
        func: Callable[..., None],  # Use ... for flexible args, will inspect signature
    ) -> Callable[[MQTTMessage], None]:
        @wraps(func)
        def wrapper(msg: MQTTMessage) -> None:
            topic = msg.topic
            parsed_message: Optional[AnyMessage] = None

            try:
                # 1. Parse and Validate using Pydantic helper
                parsed_message = from_mqtt_message(msg)
                # Log successful parsing
                messager.log(
                    f"Decorator: Parsed {type(parsed_message).__name__} from topic {topic}",
                    level="debug",
                )

            except (ValueError, ValidationError) as e:
                # Handle errors during parsing/validation
                payload_preview = msg.payload[:100].decode("utf-8", errors="replace")
                err_msg = f"Invalid/unparseable message on {topic}. Error: {e}. Payload preview: '{payload_preview}...'"
                print(err_msg)  # Also print for immediate visibility
                messager.log(err_msg, level="error")
                return  # Stop processing if message is invalid

            except Exception as e:
                # Catch unexpected errors during parsing phase
                payload_preview = msg.payload[:100].decode("utf-8", errors="replace")
                err_msg = f"Unexpected error parsing message on {topic}: {e}. Payload preview: '{payload_preview}...'"
                print(err_msg)
                messager.log(err_msg, level="error")
                import traceback

                traceback.print_exc()
                return  # Stop processing

            # 2. Call original handler with injected dependencies and kwargs
            try:
                sig = inspect.signature(func)
                params = sig.parameters
                call_kwargs = {}

                # Prepare kwargs based on available parameters in the function signature
                if "messager" in params:
                    call_kwargs["messager"] = messager
                # Inject the parsed Pydantic message object
                # Use a common name like 'message' or 'parsed_msg' in handlers
                if "message" in params:  # Assuming handlers use 'message' for the parsed object
                    call_kwargs["message"] = parsed_message
                elif "parsed_msg" in params:  # Alternative name
                    call_kwargs["parsed_msg"] = parsed_message
                # Inject the original MQTT message if needed
                if "mqtt_msg" in params:  # Use a distinct name like 'mqtt_msg'
                    call_kwargs["mqtt_msg"] = msg
                # Conditionally pass handler_kwargs
                if "handler_kwargs" in params:
                    call_kwargs["handler_kwargs"] = handler_kwargs
                # Add optional dependencies
                if "agent" in params and agent is not None:
                    call_kwargs["agent"] = agent

                # --- DEBUGGING ---
                # print(f"DEBUG: Decorator wrapping func: {func.__name__}")
                # print(f"DEBUG: Inspected params: {list(params.keys())}")
                # print(f"DEBUG: Call kwargs being passed: {list(call_kwargs.keys())}")
                # print(f"DEBUG: Parsed message type: {type(parsed_message).__name__}")
                # --- END DEBUGGING ---

                # Call the function with the collected kwargs
                func(**call_kwargs)

            except Exception as e:
                # Catch-all for unexpected errors within the handler logic itself
                err_msg = f"Unhandled error in handler '{func.__name__}' for topic {topic}: {e.__class__.__name__}: {e}"
                print(err_msg)
                messager.log(err_msg, level="error")
                import traceback

                traceback.print_exc()  # Print stack trace for debugging

        return wrapper

    return decorator
