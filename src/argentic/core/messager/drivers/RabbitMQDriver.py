from argentic.core.messager.drivers import BaseDriver, DriverConfig, MessageHandler
from argentic.core.protocol.message import BaseMessage

from typing import Optional, List, Dict, Any

try:
    import aio_pika

    AIO_PIKA_INSTALLED = True
except ImportError:
    AIO_PIKA_INSTALLED = False
    # Define dummy types for type hinting
    aio_pika = type(
        "aio_pika",
        (object,),
        {
            "RobustConnection": type("RobustConnection", (object,), {}),
            "Channel": type("Channel", (object,), {}),
            "ExchangeType": type("ExchangeType", (object,), {"FANOUT": "fanout"}),
            "Message": type("Message", (object,), {}),
            "IncomingMessage": type("IncomingMessage", (object,), {}),
            "Queue": type("Queue", (object,), {}),
            "connect_robust": lambda _: None,  # Dummy function
        },
    )

import json
import logging


class RabbitMQDriver(BaseDriver):
    def __init__(self, config: DriverConfig):
        if not AIO_PIKA_INSTALLED:
            raise ImportError(
                "aio-pika is not installed. "
                "Please install it with: uv pip install argentic[rabbitmq]"
            )
        super().__init__(config)
        self._connection: Optional[aio_pika.RobustConnection] = None
        self._channel: Optional[aio_pika.Channel] = None
        # topic to list of handlers
        self._listeners: Dict[str, List[MessageHandler]] = {}
        # track queues per topic
        self._queues: Dict[str, aio_pika.Queue] = {}
        self.logger = logging.getLogger("RabbitMQDriver")

    async def connect(self) -> None:
        # Determine virtualhost, defaulting to '/' if not specified or empty
        virtualhost = getattr(self.config, "virtualhost", None) or "/"
        # Ensure virtualhost starts with a slash if it's not empty and doesn't have one
        if virtualhost != "/" and not virtualhost.startswith("/"):
            virtualhost = "/" + virtualhost
        # If virtualhost is just '/', ensure no double slash in the URL if user was empty and we defaulted
        if virtualhost == "/":
            url_vhost_part = ""  # aio_pika implicitly uses / if path is empty
        else:
            url_vhost_part = virtualhost

        url = f"amqp://{self.config.user}:{self.config.password}@{self.config.url}:{self.config.port}{url_vhost_part}"
        self.logger.info(
            f"Connecting to RabbitMQ with URL: {url}"
        )  # Log the full URL for debugging
        self._connection = await aio_pika.connect_robust(url)
        self._channel = await self._connection.channel()
        self.logger.info(
            f"Connected to RabbitMQ at {self.config.url}:{self.config.port}, vhost: {virtualhost}"
        )

    async def disconnect(self) -> None:
        self.logger.info("Attempting to disconnect from RabbitMQ...")
        if self._connection:
            self.logger.info(
                f"Connection object exists. is_closed: {getattr(self._connection, 'is_closed', 'N/A')}"
            )
            try:
                await self._connection.close()
                self.logger.info("Successfully closed RabbitMQ connection.")
            except Exception as e:
                self.logger.error(f"Error during RabbitMQ connection.close(): {e!r}", exc_info=True)
        else:
            self.logger.info("No active RabbitMQ connection object to close.")
        # Reset state
        self._connection = None
        self._channel = None
        self._listeners = {}
        self._queues = {}
        self.logger.info("RabbitMQ driver state reset after disconnect.")

    async def publish(self, topic: str, payload: Any, **kwargs) -> None:
        try:
            exchange = await self._channel.declare_exchange(topic, aio_pika.ExchangeType.FANOUT)

            # Handle BaseMessage serialization with multiple fallback methods
            if isinstance(payload, BaseMessage):
                try:
                    body = payload.model_dump_json().encode("utf-8")
                except Exception as e:
                    self.logger.warning(f"Failed to serialize with model_dump_json: {e}")
                    try:
                        body = json.dumps(payload.model_dump()).encode("utf-8")
                    except Exception as e:
                        self.logger.warning(f"Failed to serialize with model_dump: {e}")
                        # Last resort serialization
                        body = json.dumps(
                            {
                                "id": str(payload.id),
                                "timestamp": (
                                    payload.timestamp.isoformat() if payload.timestamp else None
                                ),
                                "type": payload.__class__.__name__,
                                **{
                                    k: v
                                    for k, v in payload.__dict__.items()
                                    if not k.startswith("_")
                                },
                            }
                        ).encode("utf-8")
            elif isinstance(payload, str):
                body = payload.encode("utf-8")
            elif isinstance(payload, bytes):
                body = payload
            else:
                body = json.dumps(payload).encode("utf-8")

            message = aio_pika.Message(body=body)
            await exchange.publish(message, routing_key="")
            self.logger.debug(f"Published message to exchange: {topic}")
        except Exception as e:
            self.logger.error(f"Error publishing to exchange {topic}: {e}")
            raise

    async def subscribe(self, topic: str, handler: MessageHandler, **kwargs) -> None:
        try:
            self.logger.info(f"Subscribe called for topic: {topic}")
            if not self._channel:
                self.logger.error(f"Cannot subscribe to topic '{topic}', channel is not available.")
                # Or raise an exception if this state is unexpected
                raise RuntimeError("RabbitMQ channel not available for subscription.")

            # register handler and setup consumer on first subscribe per topic
            if topic not in self._listeners:
                self._listeners[topic] = []
                self.logger.info(
                    f"First subscription for topic '{topic}', setting up exchange and queue."
                )
                # declare exchange and queue
                exchange = await self._channel.declare_exchange(topic, aio_pika.ExchangeType.FANOUT)
                self.logger.debug(f"Declared exchange '{topic}' type FANOUT.")
                queue = await self._channel.declare_queue(exclusive=True)
                self.logger.debug(f"Declared exclusive queue '{queue.name}' for topic '{topic}'.")
                await queue.bind(exchange)
                self.logger.debug(f"Bound queue '{queue.name}' to exchange '{topic}'.")
                self._queues[topic] = queue
                self.logger.info(f"Created and bound queue '{queue.name}' for topic: {topic}")

                # single reader for this topic
                async def _reader(message: aio_pika.IncomingMessage) -> None:
                    self.logger.debug(
                        f"[_reader for {topic}] Received raw message. Message ID: {message.message_id}, Correlation ID: {message.correlation_id}, Routing Key: {message.routing_key}"
                    )
                    try:
                        self.logger.debug(
                            f"[_reader for {topic}] Entering message.process() context for message ID: {message.message_id}"
                        )
                        async with message.process():
                            self.logger.debug(
                                f"[_reader for {topic}] Successfully entered message.process() for message ID: {message.message_id}. Body type: {type(message.body)}, Body (first 100 bytes): {message.body[:100]!r}"
                            )
                            topic_handlers = self._listeners.get(topic, [])
                            if not topic_handlers:
                                self.logger.warning(
                                    f"[_reader for {topic}] No handlers found for message ID: {message.message_id} after entering process context. This should not happen if consumer is active."
                                )
                                return

                            for i, h in enumerate(topic_handlers):
                                handler_name = getattr(h, "__name__", str(h))
                                self.logger.debug(
                                    f"[_reader for {topic}] Invoking handler {i+1}/{len(topic_handlers)} ('{handler_name}') for message ID: {message.message_id}"
                                )
                                try:
                                    await h(message.body)
                                    self.logger.debug(
                                        f"[_reader for {topic}] Handler '{handler_name}' completed for message ID: {message.message_id}"
                                    )
                                except Exception as e_handler:
                                    self.logger.error(
                                        f"[_reader for {topic}] Handler '{handler_name}' failed for message ID: {message.message_id}: {e_handler!r}",
                                        exc_info=True,
                                    )
                        self.logger.debug(
                            f"[_reader for {topic}] Exited message.process() context for message ID: {message.message_id}"
                        )
                    except Exception as e_process:
                        self.logger.error(
                            f"[_reader for {topic}] Exception during message.process() or handler invocation for message ID: {message.message_id}: {e_process!r}",
                            exc_info=True,
                        )
                        # Depending on aio_pika's behavior, we might need to nack or requeue explicitly if message.process() fails internally
                        # For now, logging the error. If message.process() raises, it typically means the message wasn't acked.

                # Start consumer
                self.logger.info(f"Starting consumer for queue '{queue.name}' on topic '{topic}'.")
                consumer_tag = await queue.consume(
                    _reader
                )  # Store consumer_tag if needed for cancellation
                self.logger.info(
                    f"Started consumer for queue '{queue.name}' (topic: {topic}) with consumer_tag: {consumer_tag}."
                )
                # Store consumer_tag in self._queues or a new dict if you need to cancel specific consumers later
                # For example: self._queues[topic] = {'queue': queue, 'consumer_tag': consumer_tag}

            self._listeners[topic].append(handler)
            self.logger.info(
                f"Added handler for topic: {topic}, total handlers: {len(self._listeners[topic])}"
            )
        except Exception as e:
            self.logger.error(f"Error subscribing to topic {topic}: {e}")
            raise

    def is_connected(self) -> bool:
        return bool(self._connection and not getattr(self._connection, "is_closed", True))

    def format_connection_error_details(self, exception: Exception) -> Optional[str]:
        """Extracts RabbitMQ specific Connection.Close frame details from an exception."""
        details = []
        details.append(f"Exception type: {type(exception)}")
        if hasattr(exception, "args") and exception.args:
            details.append(f"Exception args: {exception.args!r}")
            # The Connection.Close object is often in e.args[1] for specific aio_pika errors
            if (
                len(exception.args) > 1
                and hasattr(exception.args[1], "reply_code")
                and hasattr(exception.args[1], "reply_text")
            ):
                close_frame = exception.args[1]
                details.append(
                    f"RabbitMQ Connection.Close frame (from e.args[1]): "
                    f"reply_code={close_frame.reply_code}, reply_text='{close_frame.reply_text}'"
                )
                return "\n".join(details)
            # Handle cases where the frame might be an attribute of the exception itself
            # (e.g. aio_pika.exceptions.ProbableAuthenticationError might have a 'frame' attribute)
            elif (
                hasattr(exception, "frame")
                and hasattr(exception.frame, "reply_code")
                and hasattr(exception.frame, "reply_text")
            ):
                close_frame = exception.frame
                details.append(
                    f"RabbitMQ Connection.Close frame (from e.frame): "
                    f"reply_code={close_frame.reply_code}, reply_text='{close_frame.reply_text}'"
                )
                return "\n".join(details)

        # If specific details were found and returned, this part is skipped.
        # If not, but we still have some details, return them.
        if len(details) > 1:  # More than just the type was added
            return "\n".join(details)

        return None  # No specific details extracted or only type was available
