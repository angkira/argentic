import json
from functools import wraps
from typing import Callable, Dict, Any, Optional
from core.messager import (
    Messager,
    MQTTMessage,
)  # Assuming Messager and MQTTMessage are importable


def mqtt_handler_decorator(messager: Messager) -> Callable:
    """
    Decorator for MQTT message handlers to handle common boilerplate:
    - Decode payload (UTF-8)
    - Parse JSON payload
    - Basic error handling and logging for decoding/parsing errors.
    """

    def decorator(
        func: Callable[[Dict[str, Any], MQTTMessage], None],
    ) -> Callable[[MQTTMessage], None]:
        @wraps(func)
        def wrapper(msg: MQTTMessage) -> None:
            topic = msg.topic
            payload_str: Optional[str] = None
            data: Optional[Dict[str, Any]] = None

            try:
                # 1. Decode Payload
                payload_str = msg.payload.decode("utf-8")
                print(f"MQTT Received on {topic}: {payload_str}")  # Keep basic log
                messager.publish_log(f"Received message on {topic}")

                # 2. Parse JSON
                try:
                    data = json.loads(payload_str)
                except json.JSONDecodeError:
                    err_msg = f"Invalid JSON received on {topic}: {payload_str}"
                    print(err_msg)
                    messager.publish_log(err_msg, level="error")
                    # Optionally publish an error status message here if needed
                    return  # Stop processing if JSON is invalid

                # 3. Call original handler with parsed data and original message
                func(data, msg)

            except UnicodeDecodeError:
                err_msg = f"Cannot decode payload on {topic} as UTF-8: {msg.payload!r}"
                print(err_msg)
                messager.publish_log(err_msg, level="error")
                # Optionally publish an error status message here if needed
            except Exception as e:
                # Catch-all for unexpected errors within the handler logic itself
                # or during the initial processing steps.
                err_msg = f"Unhandled error processing message on {topic}: {e}"
                print(err_msg)
                messager.publish_log(err_msg, level="error")
                # Optionally publish an error status message here if needed

        return wrapper

    return decorator
