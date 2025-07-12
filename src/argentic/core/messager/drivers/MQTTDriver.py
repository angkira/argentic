import asyncio
from typing import Optional
from contextlib import AsyncExitStack

import aiomqtt
from aiomqtt import Client, MqttError, Message

from argentic.core.protocol.message import BaseMessage
from argentic.core.logger import get_logger, LogLevel
from argentic.core.messager.drivers.base_definitions import BaseDriver, DriverConfig, MessageHandler

logger = get_logger("mqtt_driver", LogLevel.INFO)


class MQTTDriver(BaseDriver):
    def __init__(self, config: DriverConfig):
        super().__init__(config)
        self._client: Optional[Client] = None
        self._connected = False
        self._subscriptions: dict[str, MessageHandler] = {}
        self._message_task: Optional[asyncio.Task] = None
        self._stack: Optional[AsyncExitStack] = None

    async def connect(self) -> bool:
        try:
            self._stack = AsyncExitStack()

            # Create aiomqtt client
            self._client = Client(
                hostname=self.config.url,
                port=self.config.port,
                username=self.config.user,
                password=self.config.password,
                identifier=self.config.client_id,
                keepalive=self.config.keepalive or 60,
            )

            # Connect using the async context manager
            await self._stack.enter_async_context(self._client)

            # Start message handler task
            self._message_task = asyncio.create_task(self._handle_messages())

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

            # Cancel message handler task
            if self._message_task and not self._message_task.done():
                self._message_task.cancel()
                try:
                    await self._message_task
                except asyncio.CancelledError:
                    pass

            # Close the client context
            if self._stack:
                await self._stack.aclose()
                self._stack = None

            self._client = None
            logger.info("MQTT disconnected.")

    def is_connected(self) -> bool:
        return self._connected

    async def publish(self, topic: str, payload: BaseMessage, **kwargs) -> None:
        if not self._connected or not self._client:
            raise ConnectionError("Not connected to MQTT broker.")

        try:
            # Use aiomqtt's publish method with proper JSON serialization
            json_payload = payload.model_dump_json()
            await self._client.publish(
                topic=topic,
                payload=json_payload,
                qos=kwargs.get("qos", 1),
                retain=kwargs.get("retain", False),
            )
            logger.debug(f"Published message to topic: {topic}")

        except Exception as e:
            logger.error(f"Error publishing message to {topic}: {e}")
            raise

    async def subscribe(self, topic: str, handler: MessageHandler, **kwargs) -> None:
        if not self._connected or not self._client:
            raise ConnectionError("Not connected to MQTT broker.")

        try:
            # Store the handler for this topic
            self._subscriptions[topic] = handler

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
            # Remove the handler
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
                await self._process_message(message)
        except asyncio.CancelledError:
            logger.debug("Message handler task cancelled")
        except Exception as e:
            logger.error(f"Error in message handler: {e}")

    async def _process_message(self, message: Message) -> None:
        """Process a single message from aiomqtt."""
        try:
            # Find handler for this topic
            handler = self._subscriptions.get(message.topic.value)
            if not handler:
                logger.warning(f"No handler for topic {message.topic.value}")
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

                # Parse as BaseMessage but store the original JSON for re-parsing
                base_message = BaseMessage.model_validate_json(payload_str)
                # Store the original JSON string as an extra attribute for the handler
                setattr(base_message, "_original_json", payload_str)

            except Exception as e:
                logger.error(f"Failed to parse message from {message.topic.value}: {e}")
                return

            # Call the handler
            await handler(base_message)

        except Exception as e:
            logger.error(f"Error processing message from {message.topic.value}: {e}")

    def format_connection_error_details(self, error: Exception) -> Optional[str]:
        """Format MQTT-specific connection error details."""
        if isinstance(error, MqttError):
            return f"MQTT error: {error}"
        return None
