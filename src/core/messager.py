import asyncio
import time
import json
import traceback
from typing import Callable, Dict, Optional, Union, Coroutine, Any
import ssl

import aiomqtt

# Import the correct function from paho.mqtt.client
from paho.mqtt.client import topic_matches_sub

from core.logger import get_logger, LogLevel, parse_log_level
from core.protocol.message import BaseMessage, from_payload

# Define MessageHandler type hint
MessageHandler = Callable[[aiomqtt.Message], Coroutine[Any, Any, None]]

# Add type alias for backward compatibility with existing code
MQTTMessage = aiomqtt.Message


class Messager:
    """Async MQTT client"""

    def __init__(
        self,
        broker_address: str,
        port: int = 1883,
        client_id: str = "",
        username: Optional[str] = None,
        password: Optional[str] = None,
        keepalive: int = 60,
        pub_log_topic: Optional[str] = None,
        log_level: Union[LogLevel, str] = LogLevel.INFO,
        tls_params: Optional[Dict[str, Any]] = None,
    ):
        self.broker_address = broker_address
        self.port = port
        self.client_id = client_id or f"client-{int(time.time())}"
        self.username = username
        self.password = password
        self.keepalive = keepalive
        self.pub_log_topic = pub_log_topic

        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level

        # Use client_id in logger name for clarity if multiple clients run
        self.logger = get_logger(f"mqtt.{self.client_id}", level=self.log_level)

        self._client: Optional[aiomqtt.Client] = None
        self._topic_handlers: Dict[str, MessageHandler] = {}  # Topic pattern -> Handler Coroutine
        self._listen_task: Optional[asyncio.Task] = None
        self._connection_event = asyncio.Event()
        self._is_disconnecting = False  # Flag to prevent reconnect during shutdown

        # Configure TLS if parameters are provided
        self._tls_params = None
        if tls_params:
            try:
                self._tls_params = aiomqtt.TLSParameters(
                    ca_certs=tls_params.get("ca_certs"),
                    certfile=tls_params.get("certfile"),
                    keyfile=tls_params.get("keyfile"),
                    cert_reqs=getattr(
                        ssl, tls_params.get("cert_reqs", "CERT_REQUIRED"), ssl.CERT_REQUIRED
                    ),
                    tls_version=getattr(
                        ssl, tls_params.get("tls_version", "PROTOCOL_TLS"), ssl.PROTOCOL_TLS
                    ),
                    ciphers=tls_params.get("ciphers"),
                )
                self.logger.info("TLS parameters configured.")
            except Exception as e:
                self.logger.error(f"Failed to configure TLS parameters: {e}", exc_info=True)
                raise ValueError(f"Invalid TLS configuration: {e}") from e

    def is_connected(self) -> bool:
        """Check if the client is currently connected."""
        return self._client is not None and self._client._connected

    async def _handle_message(self, mqtt_message: aiomqtt.Message):
        """Internal callback to dispatch messages to registered handlers."""
        topic = mqtt_message.topic.value
        payload = mqtt_message.payload
        self.logger.debug(f"Received message on topic '{topic}'")

        matched = False
        for pattern, handler in list(self._topic_handlers.items()):
            if topic_matches_sub(pattern, topic):
                matched = True
                try:
                    # Parse to protocol message
                    protocol_message = from_payload(topic, payload)
                    # Call handler with protocol message only
                    asyncio.create_task(handler(protocol_message))
                except Exception as e:
                    self.logger.error(
                        f"Error scheduling/calling handler for topic '{topic}': {e}",
                        exc_info=True,
                    )
        if not matched:
            self.logger.debug(f"No handler matched for topic '{topic}'.")

    async def _listen_for_messages(self):
        """Background task to listen for incoming messages."""
        if not self._client:
            self.logger.error("Listener task started but client is not initialized.")
            return

        self.logger.info("Message listener task started.")
        while True:
            try:
                async for message in self._client.messages:
                    await self._handle_message(message)
            except aiomqtt.MqttError as error:
                if self._is_disconnecting:
                    self.logger.info("Listener task stopping due to intentional disconnect.")
                    break

                self.logger.warning(
                    f"Listener task interrupted by MQTT error: {error}. Attempting to reconnect..."
                )
                await asyncio.sleep(5)
                if not self.is_connected():
                    self.logger.warning(
                        "Listener task: Client is disconnected. Waiting for reconnection."
                    )
                    await self._connection_event.wait()
                    self.logger.info("Listener task: Reconnected. Resuming message listening.")
                else:
                    self.logger.info(
                        "Listener task: Client reconnected. Resuming message listening."
                    )

            except asyncio.CancelledError:
                self.logger.info("Message listener task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in message listener task: {e}", exc_info=True)
                await self.log(f"Unexpected listener error: {e}", level="error")
                await asyncio.sleep(5)

        self.logger.info("Message listener task finished.")

    async def connect(self) -> bool:
        """Connects to the MQTT broker and starts the listener task."""
        if self.is_connected():
            self.logger.info("Already connected.")
            return True
        if self._is_disconnecting:
            self.logger.warning("Connection attempt aborted: Disconnection in progress.")
            return False

        self._connection_event.clear()
        self.logger.info(f"Connecting to {self.broker_address}:{self.port} as {self.client_id}...")

        try:
            if self._client is None:
                # Create a new client instance
                self._client = aiomqtt.Client(
                    hostname=self.broker_address,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    keepalive=self.keepalive,
                    logger=self.logger,
                    tls_params=self._tls_params,
                )

                # Start the client by entering its context
                await self._client.__aenter__()
                self.logger.info("MQTT client connected successfully.")
                self._connection_event.set()

                if self._listen_task is None or self._listen_task.done():
                    self._listen_task = asyncio.create_task(self._listen_for_messages())
                    self.logger.info("Started background message listener task.")

                if self._topic_handlers:
                    self.logger.info("Resubscribing to topics...")
                    for topic, handler in self._topic_handlers.items():
                        await self.subscribe(topic, handler, is_reconnect=True)

                return True

        except aiomqtt.MqttError as error:
            self.logger.error(f"Failed to connect to MQTT broker: {error}")
            self._connection_event.clear()
            self._client = None
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during connection: {e}", exc_info=True)
            self._connection_event.clear()
            self._client = None
            return False

    async def disconnect(self) -> None:
        """Disconnects from the MQTT broker gracefully."""
        if not self.is_connected() and not self._listen_task:
            self.logger.info("Already disconnected.")
            return

        self.logger.info("Disconnecting...")
        self._is_disconnecting = True
        self._connection_event.clear()

        # Cancel listener task if running
        if self._listen_task and not self._listen_task.done():
            self.logger.debug("Cancelling message listener task...")
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                self.logger.debug("Message listener task cancellation confirmed.")
            except Exception as e:
                self.logger.error(f"Error during listener task cancellation: {e}", exc_info=True)

        self._listen_task = None

        # Close the client context
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
                self.logger.info("MQTT client disconnected.")
            except Exception as e:
                self.logger.error(f"Unexpected error during disconnect: {e}", exc_info=True)
            finally:
                self._client = None

        self._is_disconnecting = False
        self.logger.info("Disconnect process complete.")

    async def publish(
        self, topic: str, payload: Union[str, bytes, Dict, list], qos: int = 0, retain: bool = False
    ) -> None:
        """Publishes a message to a topic."""
        if not self.is_connected():
            self.logger.error(f"Cannot publish to '{topic}': Not connected.")
            raise ConnectionError("MQTT client not connected")

        if isinstance(payload, (dict, list)):
            payload_bytes = json.dumps(payload).encode("utf-8")
            self.logger.debug(f"Serialized dict/list payload to JSON for topic '{topic}'")
        elif isinstance(payload, str):
            payload_bytes = payload.encode("utf-8")
        elif isinstance(payload, bytes):
            payload_bytes = payload
        else:
            self.logger.error(
                f"Invalid payload type for publish: {type(payload)}. Must be str, bytes, dict, or list."
            )
            raise TypeError("Payload must be str, bytes, dict, or list")

        try:
            await self._client.publish(topic, payload=payload_bytes, qos=qos, retain=retain)
            self.logger.debug(f"Published to topic '{topic}' (qos={qos}, retain={retain})")
        except aiomqtt.MqttError as error:
            self.logger.error(f"Failed to publish to topic '{topic}': {error}")
            raise error
        except Exception as e:
            self.logger.error(f"Unexpected error publishing to topic '{topic}': {e}", exc_info=True)
            raise e

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[BaseMessage], Coroutine],
        qos: int = 0,
        is_reconnect: bool = False,
    ) -> None:
        """Subscribes to a topic and registers a handler coroutine.
        Handler must accept a single protocol message argument (not MQTT message).
        """
        if not callable(handler) or not asyncio.iscoroutinefunction(handler):
            raise TypeError(f"Handler for topic '{topic}' must be a callable coroutine function.")

        if not self.is_connected():
            self.logger.warning(
                f"Not connected. Registering handler for '{topic}', subscription will occur upon connection."
            )
            self._topic_handlers[topic] = handler
            return

        try:
            await self._client.subscribe(topic, qos=qos)
            self._topic_handlers[topic] = handler
            handler_name = getattr(handler, "__name__", str(type(handler)))
            if not is_reconnect:
                self.logger.info(
                    f"Subscribed to topic '{topic}' (qos={qos}) with handler '{handler_name}'"
                )
            else:
                self.logger.debug(f"Resubscribed to topic '{topic}' (qos={qos})")
        except aiomqtt.MqttError as error:
            self.logger.error(f"Failed to subscribe to topic '{topic}': {error}")
            raise error
        except Exception as e:
            self.logger.error(
                f"Unexpected error subscribing to topic '{topic}': {e}", exc_info=True
            )
            raise e

    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribes from a topic and removes the handler."""
        if topic in self._topic_handlers:
            del self._topic_handlers[topic]
            self.logger.debug(f"Removed handler for topic '{topic}'.")
        else:
            self.logger.warning(f"No handler registered for topic '{topic}' to remove.")

        if not self.is_connected():
            self.logger.warning(
                f"Not connected. Cannot unsubscribe from broker topic '{topic}' now."
            )
            return

        try:
            await self._client.unsubscribe(topic)
            self.logger.info(f"Unsubscribed from topic '{topic}'.")
        except aiomqtt.MqttError as error:
            self.logger.error(f"Failed to unsubscribe from topic '{topic}': {error}")
        except Exception as e:
            self.logger.error(
                f"Unexpected error unsubscribing from topic '{topic}': {e}", exc_info=True
            )

    async def log(self, message: str, level: str = "info") -> None:
        """
        Publishes a log message to the configured log topic.

        Args:
            message: The log message text
            level: Log level (info, debug, warning, error, critical)
        """
        if not self.pub_log_topic:
            self.logger.debug(f"Log message not sent (no pub_log_topic): [{level}] {message}")
            return

        try:
            log_payload = {
                "timestamp": time.time(),
                "level": level,
                "source": self.client_id,
                "message": message,
            }

            # Only attempt to publish if connected
            if self.is_connected():
                await self.publish(self.pub_log_topic, log_payload)
            else:
                self.logger.debug(f"Couldn't send log to MQTT (not connected): [{level}] {message}")
        except Exception as e:
            self.logger.error(f"Failed to publish log message: {e}", exc_info=True)

    async def stop(self) -> None:
        """
        Stops the messager - disconnects from broker and cleans up resources.
        This is an alias for disconnect() to provide a consistent interface.
        """
        await self.disconnect()
