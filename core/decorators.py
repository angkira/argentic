import json
from functools import wraps
from typing import Callable, Dict, Any, Optional

# Local application imports (assuming these are still correct relative paths)
from core.messager import Messager, MQTTMessage
from core.rag import RAGController  # Import RAGController


# Modified decorator factory
def mqtt_handler_decorator(
    messager: Messager, rag_controller: Optional[RAGController] = None, **handler_kwargs
) -> Callable:
    """
    Decorator factory for MQTT message handlers.

    Handles common boilerplate and injects dependencies.
    - Decodes payload (UTF-8)
    - Parses JSON payload
    - Basic error handling for decoding/parsing.
    - Injects Messager, RAGController (optional), parsed data, original message,
      and any extra keyword arguments passed to the decorator factory
      into the decorated handler function.
    """

    def decorator(
        # The decorated function now expects injected args + handler_kwargs
        func: Callable[
            [
                Messager,
                Optional[RAGController],
                Optional[Dict[str, Any]],
                MQTTMessage,
                Dict[str, Any],  # handler_kwargs
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
                    messager.publish_log(err_msg, level="warning")
                    data = None  # Pass None for data if JSON parsing fails

                # 3. Call original handler with injected dependencies and kwargs
                func(messager, rag_controller, data, msg, handler_kwargs)

            except UnicodeDecodeError:
                err_msg = f"Cannot decode payload on {topic} as UTF-8: {msg.payload!r}"
                print(err_msg)
                messager.publish_log(err_msg, level="error")
            except Exception as e:
                # Catch-all for unexpected errors within the handler logic itself
                err_msg = f"Unhandled error processing message on {topic}: {e.__class__.__name__}: {e}"
                print(err_msg)
                messager.publish_log(err_msg, level="error")
                # Optionally re-raise or handle more gracefully depending on needs
                # raise # Uncomment to propagate the error if needed

        return wrapper

    return decorator
