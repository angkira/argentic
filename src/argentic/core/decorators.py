from functools import wraps
from typing import Callable, Any, Optional, Coroutine
import inspect

from argentic.core.messager.messager import Messager
from argentic.core.agent.agent import Agent
from argentic.core.protocol.message import BaseMessage


def mqtt_handler_decorator(
    messager: Messager,
    agent: Optional[Agent] = None,
    topic: str = "",
    **handler_kwargs,
) -> Callable:
    """
    Decorator factory for MQTT message handlers using Pydantic.

    Creates a MessageHandler function compatible with Messager.subscribe()
    that will inject dependencies into your handler function.

    Args:
        messager: The Messager instance to use for logging
        agent: Optional Agent to inject into handlers
        topic: Topic name for logging purposes
        **handler_kwargs: Additional keyword arguments to pass to the handler

    Returns:
        A decorator that wraps an async function to be used as a message handler
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, None]],
    ) -> Callable[[BaseMessage], Coroutine[Any, Any, None]]:
        """
        Decorator that wraps an async function to handle parsed messages.
        The wrapped function will be called with dependencies injected based
        on its parameter names.
        """

        @wraps(func)
        async def wrapper(message: BaseMessage) -> None:
            """
            Wrapper function that handles dependency injection.
            This signature is compatible with MessageHandler from drivers/__init__.py
            """
            try:
                # Inspect the signature of the async function
                sig = inspect.signature(func)
                params = sig.parameters
                call_kwargs = {}

                # Prepare kwargs based on available parameters
                if "messager" in params:
                    call_kwargs["messager"] = messager
                # Inject the message object
                if "message" in params:
                    call_kwargs["message"] = message
                # Conditionally pass handler_kwargs
                if "handler_kwargs" in params:
                    call_kwargs["handler_kwargs"] = handler_kwargs
                # Add optional dependencies
                if "agent" in params and agent is not None:
                    call_kwargs["agent"] = agent
                # Add topic if requested
                if "topic" in params and topic:
                    call_kwargs["topic"] = topic

                # Await the original async function call
                await func(**call_kwargs)

            except Exception as e:
                # Catch-all for unexpected errors within the handler logic
                err_msg = f"Unhandled error in handler '{func.__name__}' for topic {topic}: {e.__class__.__name__}: {e}"
                print(err_msg)
                await messager.log(err_msg, level="error")
                import traceback

                traceback.print_exc()

        return wrapper

    return decorator
