import json
import asyncio  # Import asyncio
from functools import wraps
from typing import Callable, Dict, Any, Optional, Type, Coroutine
import inspect

import aiomqtt  # Import aiomqtt for message type hint
from pydantic import ValidationError

# Local application imports
from core.messager import Messager
from core.agent import Agent  # Keep if Agent injection is needed

# Assuming from_mqtt_message is updated or replaced to handle aiomqtt.Message
from core.protocol.message import from_payload, AnyMessage, BaseMessage


# Updated async decorator factory
def mqtt_handler_decorator(
    messager: Messager,
    agent: Optional[Agent] = None,
    **handler_kwargs,
) -> Callable:
    """
    Async Decorator factory for MQTT message handlers using Pydantic and aiomqtt.

    Handles common boilerplate and injects dependencies asynchronously.
    - Parses the incoming MQTT message payload using `from_payload`.
    - Handles parsing/validation errors.
    - Injects Messager, the parsed Pydantic message object, Agent (optional),
      original aiomqtt.Message, and any extra keyword arguments passed to the
      decorator factory into the decorated async handler function.
    """

    def decorator(
        # The decorated function must now be an async function
        func: Callable[..., Coroutine[Any, Any, None]],
    ) -> Callable[[aiomqtt.Message], Coroutine[Any, Any, None]]:
        @wraps(func)
        # The wrapper itself must be async and accept aiomqtt.Message
        async def wrapper(msg: aiomqtt.Message) -> None:
            topic = msg.topic.value  # Get topic string from aiomqtt.Message
            payload = msg.payload  # Get payload bytes from aiomqtt.Message
            parsed_message: Optional[AnyMessage] = None

            try:
                # 1. Parse and Validate using Pydantic helper (assuming from_payload exists)
                # Pass topic and payload separately if needed by the parser
                parsed_message = from_payload(topic, payload)  # Adapt this call as needed

                # Log successful parsing (use await)
                await messager.log(
                    f"Decorator: Parsed {type(parsed_message).__name__} from topic {topic}",
                    level="debug",
                )

            except (ValueError, ValidationError) as e:
                # Handle errors during parsing/validation
                payload_preview = payload[:100].decode("utf-8", errors="replace")
                err_msg = f"Invalid/unparseable message on {topic}. Error: {e}. Payload preview: '{payload_preview}...'"
                print(err_msg)
                # Use await for logging
                await messager.log(err_msg, level="error")
                return

            except Exception as e:
                # Catch unexpected errors during parsing phase
                payload_preview = payload[:100].decode("utf-8", errors="replace")
                err_msg = f"Unexpected error parsing message on {topic}: {e}. Payload preview: '{payload_preview}...'"
                print(err_msg)
                # Use await for logging
                await messager.log(err_msg, level="error")
                import traceback

                traceback.print_exc()
                return

            # 2. Call original async handler with injected dependencies and kwargs
            try:
                # Inspect the signature of the async function
                sig = inspect.signature(func)
                params = sig.parameters
                call_kwargs = {}

                # Prepare kwargs based on available parameters
                if "messager" in params:
                    call_kwargs["messager"] = messager
                # Inject the parsed Pydantic message object
                if "message" in params:
                    call_kwargs["message"] = parsed_message
                elif "parsed_msg" in params:
                    call_kwargs["parsed_msg"] = parsed_message
                # Inject the original aiomqtt message if needed
                if "mqtt_msg" in params:
                    call_kwargs["mqtt_msg"] = msg
                # Conditionally pass handler_kwargs
                if "handler_kwargs" in params:
                    call_kwargs["handler_kwargs"] = handler_kwargs
                # Add optional dependencies
                if "agent" in params and agent is not None:
                    call_kwargs["agent"] = agent

                # --- DEBUGGING ---
                # print(f"DEBUG: Decorator wrapping async func: {func.__name__}")
                # print(f"DEBUG: Inspected params: {list(params.keys())}")
                # print(f"DEBUG: Call kwargs being passed: {list(call_kwargs.keys())}")
                # print(f"DEBUG: Parsed message type: {type(parsed_message).__name__}")
                # --- END DEBUGGING ---

                # Await the original async function call
                await func(**call_kwargs)

            except Exception as e:
                # Catch-all for unexpected errors within the handler logic itself
                err_msg = f"Unhandled error in async handler '{func.__name__}' for topic {topic}: {e.__class__.__name__}: {e}"
                print(err_msg)
                # Use await for logging
                await messager.log(err_msg, level="error")
                import traceback

                traceback.print_exc()

        return wrapper

    return decorator
