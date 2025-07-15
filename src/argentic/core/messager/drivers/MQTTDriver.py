import asyncio
import time
from typing import Optional, Dict, Any, cast
from contextlib import AsyncExitStack

import aiomqtt
from aiomqtt import Client, MqttError, Message
from aiomqtt.client import ProtocolVersion

from argentic.core.protocol.message import BaseMessage, InfoMessage
from argentic.core.logger import get_logger, LogLevel
from argentic.core.messager.drivers.base_definitions import BaseDriver, MessageHandler
from .configs import MQTTDriverConfig

logger = get_logger("mqtt_driver", LogLevel.DEBUG)


class MQTTDriver(BaseDriver[MQTTDriverConfig]):
    def __init__(self, config: MQTTDriverConfig):
        super().__init__(config)
        self._client: Optional[Client] = None
        self._connected = False
        # Dictionary: topic -> {message_cls_name: (handler, message_cls)}
        self._subscriptions: Dict[str, Dict[str, tuple[MessageHandler, type]]] = {}
        self._message_task: Optional[asyncio.Task] = None
        self._stack: Optional[AsyncExitStack] = None
        # EXPERIMENTAL: Task pool to prevent handler blocking
        self._handler_tasks: set = set()
        self._max_concurrent_handlers = 50
        # Shadow ping to keep connection alive. Interval = keepalive / 2
        if self.config.keepalive is not None and self.config.keepalive > 0:
            self._ping_interval: float = self.config.keepalive / 2.0
        else:
            # Fallback to 30 s if keepalive is disabled or invalid
            self._ping_interval = 30.0
        self._last_ping_time = 0.0
        self._ping_task: Optional[asyncio.Task] = None

    async def connect(self, start_ping: bool = True) -> bool:
        try:
            self._stack = AsyncExitStack()

            # Create aiomqtt client
            self._client = Client(
                hostname=self.config.url,
                port=self.config.port,
                username=self.config.user,
                password=self.config.password,
                identifier=self.config.client_id,
                keepalive=self.config.keepalive,
                protocol=self.config.version,
            )

            # Connect using the async context manager
            await self._stack.enter_async_context(self._client)

            # Start message handler task
            self._message_task = asyncio.create_task(self._handle_messages())

            # Start shadow ping task to keep connection alive
            if start_ping:
                self._ping_task = asyncio.create_task(self._shadow_ping_loop())

            self._connected = True
            logger.info("MQTT connected via aiomqtt.")
            return True

        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            self._connected = False
            if self._stack:
                await self._stack.aclose()
                self._stack = None
            return False

    async def disconnect(self) -> None:
        if self._connected:
            self._connected = False

            # Cancel ping task
            if self._ping_task and not self._ping_task.done():
                self._ping_task.cancel()
                try:
                    await self._ping_task
                except asyncio.CancelledError:
                    pass

            # Cancel message handler task
            if self._message_task and not self._message_task.done():
                self._message_task.cancel()
                try:
                    await self._message_task
                except asyncio.CancelledError:
                    pass

            # EXPERIMENTAL: Cancel all pending handler tasks
            for task in self._handler_tasks:
                if not task.done():
                    task.cancel()
            if self._handler_tasks:
                try:
                    await asyncio.gather(*self._handler_tasks, return_exceptions=True)
                except Exception as e:
                    logger.debug(f"Error cleaning up handler tasks: {e}")
            self._handler_tasks.clear()

            # Close the client context
            if self._stack:
                await self._stack.aclose()
                self._stack = None

            self._client = None
            logger.info("MQTT disconnected.")

    def is_connected(self) -> bool:
        return self._connected

    async def publish(
        self, topic: str, payload: BaseMessage, qos: int = 0, retain: bool = False
    ) -> None:
        """
        Publish a message to MQTT.

        If the publish fails due to a lost connection, the driver will
        attempt a single reconnection and retry the publish once.
        """

        attempt = 0
        while attempt < 2:  # first try + one retry after reconnect
            attempt += 1

            # If we already know we're disconnected try reconnect first
            if (not self._connected) or (self._client is None):
                logger.debug("MQTT not connected – attempting to reconnect before publish …")
                await self._reconnect()

            try:
                client = cast(Client, self._client)  # type narrowing for linters
                serialized_data = payload.model_dump_json()
                await client.publish(topic, serialized_data.encode(), qos=qos, retain=retain)
                return  # Success
            except Exception as publish_err:
                logger.warning(f"Publish attempt {attempt} to '{topic}' failed: {publish_err}")

                if attempt >= 2:
                    # No more retries – propagate the error
                    raise

                # Mark as disconnected and try to reconnect once
                self._connected = False
                await self._reconnect()

    async def _reconnect(self) -> None:
        """Attempt to re-establish the MQTT connection.

        This helper is designed to be lightweight so it can be called
        directly from the publish path. The existing shadow-ping task
        continues running; therefore we do NOT spawn a new one here.
        """

        logger.info("Attempting MQTT reconnection …")

        # Close previous connection gracefully
        try:
            if self._stack:
                await self._stack.aclose()
        except Exception:
            pass

        self._client = None
        self._connected = False

        # Re-create connection without starting another ping loop
        await self.connect(start_ping=False)

        # Re-subscribe to previously registered topics
        if self._connected and self._client:
            client = cast(Client, self._client)
            for topic in list(self._subscriptions.keys()):
                try:
                    await client.subscribe(topic, qos=1)
                    logger.debug(f"Resubscribed to topic '{topic}' after reconnect")
                except Exception as sub_err:
                    logger.warning(f"Failed to resubscribe to '{topic}' after reconnect: {sub_err}")

    async def subscribe(
        self, topic: str, handler: MessageHandler, message_cls: type = BaseMessage, **kwargs
    ) -> None:
        if not self._connected or not self._client:
            raise ConnectionError("Not connected to MQTT broker.")

        try:
            # Store the handler for this topic
            if topic not in self._subscriptions:
                self._subscriptions[topic] = {}
            self._subscriptions[topic][message_cls.__name__] = (handler, message_cls)

            # Subscribe using aiomqtt
            await self._client.subscribe(topic, qos=kwargs.get("qos", 1))
            logger.info(f"Subscribed to topic: {topic}")

        except Exception as e:
            logger.error(f"Error subscribing to {topic}: {e}")
            raise

    async def unsubscribe(self, topic: str) -> None:
        if not self._connected or not self._client:
            raise ConnectionError("Not connected to MQTT broker.")

        try:
            # Remove all handlers for this topic
            if topic in self._subscriptions:
                del self._subscriptions[topic]

            # Unsubscribe using aiomqtt
            await self._client.unsubscribe(topic)
            logger.info(f"Unsubscribed from topic: {topic}")

        except Exception as e:
            logger.error(f"Error unsubscribing from {topic}: {e}")
            raise

    async def _handle_messages(self) -> None:
        """Handle incoming messages from aiomqtt."""
        if not self._client:
            return

        try:
            async for message in self._client.messages:
                # EXPERIMENTAL: Don't await _process_message directly - spawn as task
                # This prevents any handler from blocking the main message loop
                if len(self._handler_tasks) >= self._max_concurrent_handlers:
                    logger.warning("Handler task pool full, waiting for completion...")
                    # Wait for at least one task to complete
                    done, pending = await asyncio.wait(
                        self._handler_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    # Clean up completed tasks
                    for task in done:
                        self._handler_tasks.discard(task)
                        if task.exception():
                            logger.error(f"Handler task failed: {task.exception()}")

                # Spawn message processing as independent task
                task = asyncio.create_task(self._process_message(message))
                self._handler_tasks.add(task)

                # Clean up completed tasks periodically
                done_tasks = [t for t in self._handler_tasks if t.done()]
                for task in done_tasks:
                    self._handler_tasks.discard(task)
                    if task.exception():
                        logger.error(f"Handler task failed: {task.exception()}")

        except asyncio.CancelledError:
            logger.debug("Message handler task cancelled")
            # Cancel all pending handler tasks
            for task in self._handler_tasks:
                if not task.done():
                    task.cancel()
            # Wait for all tasks to complete
            if self._handler_tasks:
                await asyncio.gather(*self._handler_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in message handler: {e}")

    async def _process_message(self, message: Message) -> None:
        """Process a single message from aiomqtt."""
        try:
            # Find handlers for this topic
            handlers = self._subscriptions.get(message.topic.value)
            if not handlers:
                logger.warning(f"No handlers for topic {message.topic.value}")
                return

            # Parse the message payload
            try:
                # Handle different payload types
                if isinstance(message.payload, bytes):
                    payload_str = message.payload.decode()
                elif isinstance(message.payload, str):
                    payload_str = message.payload
                else:
                    payload_str = str(message.payload)

                # Parse as BaseMessage first
                base_message = BaseMessage.model_validate_json(payload_str)
                # Store the original JSON string for re-parsing
                setattr(base_message, "_original_json", payload_str)

            except Exception as e:
                logger.error(f"Failed to parse message from {message.topic.value}: {e}")
                return

            # Call appropriate handlers based on message type compatibility
            for handler_cls_name, (handler, handler_cls) in handlers.items():
                try:
                    # Try to parse the message as the specific type
                    if handler_cls is BaseMessage:
                        # Generic BaseMessage handler
                        await handler(base_message)
                    else:
                        # Try to parse as specific type
                        try:
                            validate_method = getattr(handler_cls, "model_validate_json", None)
                            if validate_method:
                                specific_message = validate_method(payload_str)
                                await handler(specific_message)
                            else:
                                # Skip non-BaseMessage handlers
                                logger.debug(f"Skipping non-BaseMessage handler {handler_cls_name}")
                                continue
                        except Exception as parse_error:
                            # Skip this handler if message doesn't match type
                            logger.debug(
                                f"Message type mismatch for handler {handler_cls_name}: {parse_error}"
                            )
                            continue
                except Exception as handler_error:
                    logger.error(
                        f"Error in handler {handler_cls_name} for topic {message.topic.value}: {handler_error}"
                    )

        except Exception as e:
            logger.error(f"Error processing message from {message.topic.value}: {e}")

    async def _shadow_ping_loop(self) -> None:
        """Send periodic ping messages to keep connection alive.

        This loop runs constantly and independently of connection state.
        It will keep trying to ping even during reconnections.
        """
        logger.info(f"Starting shadow ping loop (interval: {self._ping_interval}s)")
        ping_count = 0

        try:
            while True:
                ping_count += 1
                try:
                    await self._send_ping(ping_count)
                except Exception as ping_err:
                    logger.debug(f"Shadow ping #{ping_count} failed: {ping_err}")
                await asyncio.sleep(self._ping_interval)
        except asyncio.CancelledError:
            logger.info(f"Shadow ping loop cancelled after {ping_count} pings")
        logger.info("Shadow ping loop ended")

    async def _send_ping(self, ping_count: int) -> None:
        """Send a single ping message directly via MQTT client.

        This bypasses the main publish() method to avoid interference
        with reconnection logic.
        """
        if not self._connected:
            raise ConnectionError("Not connected to MQTT broker.")

        ping_topic = f"_ping/{self.config.client_id}"
        ping_msg = InfoMessage(
            source=self.config.client_id or "mqtt_client",
            data={"ping_id": ping_count, "timestamp": time.time()},
        )
        await self.publish(ping_topic, ping_msg, qos=0, retain=False)

    def format_connection_error_details(self, error: Exception) -> Optional[str]:
        """Format MQTT-specific connection error details."""
        if isinstance(error, MqttError):
            return f"MQTT error: {error}"
        return None
