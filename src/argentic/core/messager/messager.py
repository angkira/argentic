import time
import asyncio
from typing import Dict, Optional, Union, Any
import ssl

from pydantic import ValidationError

from argentic.core.messager.drivers import create_driver, DriverConfig, MessageHandler

from argentic.core.logger import get_logger, LogLevel, parse_log_level
from argentic.core.messager.protocols import MessagerProtocol
from argentic.core.protocol.message import BaseMessage


class Messager:
    """Asynchronous messaging client that supports multiple protocols.

    This class provides a unified interface for messaging operations including
    publishing, subscribing, and handling messages. It supports different messaging
    protocols (default is MQTT) through pluggable drivers and includes features for:

    - Establishing secure connections with TLS/SSL support
    - Publishing and subscribing to topics
    - Message type validation through Pydantic models
    - Integrated logging with configurable levels
    - Asynchronous message handling

    The client handles reconnection, message parsing, and provides a consistent
    API regardless of the underlying protocol implementation.
    """

    def __init__(
        self,
        broker_address: str,
        port: int = 1883,
        protocol: MessagerProtocol = MessagerProtocol.MQTT,
        client_id: str = "",
        username: Optional[str] = None,
        password: Optional[str] = None,
        keepalive: int = 60,
        pub_log_topic: Optional[str] = None,
        log_level: Union[LogLevel, str] = LogLevel.INFO,
        tls_params: Optional[Dict[str, Any]] = None,
        **driver_kwargs: Any,
    ):
        """Initialize a new Messager instance.

        Args:
            broker_address: Address of the message broker
            port: Broker port number
            protocol: Messaging protocol to use
            client_id: Unique client identifier, generated from timestamp if not provided
            username: Authentication username
            password: Authentication password
            keepalive: Keepalive interval in seconds
            pub_log_topic: Topic to publish log messages to, if any
            log_level: Logging level
            tls_params: TLS/SSL configuration parameters
            **driver_kwargs: Additional keyword arguments for the specific driver config
        """
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
        self.logger = get_logger(f"messager.{self.client_id}", level=self.log_level)

        self._tls_params = None
        if tls_params:
            try:
                self._tls_params = {
                    "ca_certs": tls_params.get("ca_certs"),
                    "certfile": tls_params.get("certfile"),
                    "keyfile": tls_params.get("keyfile"),
                    "cert_reqs": getattr(
                        ssl, tls_params.get("cert_reqs", "CERT_REQUIRED"), ssl.CERT_REQUIRED
                    ),
                    "tls_version": getattr(
                        ssl, tls_params.get("tls_version", "PROTOCOL_TLS"), ssl.PROTOCOL_TLS
                    ),
                    "ciphers": tls_params.get("ciphers"),
                }
                self.logger.info("TLS parameters configured.")
            except Exception as e:
                self.logger.error(f"Failed to configure TLS parameters: {e}", exc_info=True)
                raise ValueError(f"Invalid TLS configuration: {e}") from e

        # Instantiate protocol driver
        cfg = DriverConfig(
            url=broker_address,
            port=port,
            user=username,
            password=password,
            token=None,
            client_id=self.client_id,
            **driver_kwargs,
        )

        self._driver = create_driver(protocol, cfg)

    def is_connected(self) -> bool:
        """Check if the client is currently connected.

        Returns:
            bool: True if connected, False otherwise
        """
        return self._driver.is_connected()

    async def connect(self) -> bool:
        """Connect to the message broker using the configured driver.

        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            await self._driver.connect()
            self.logger.info("Connected successfully via driver")
            return True
        except Exception as e:
            log_msg = f"Driver connection failed: {e!r}"
            # Attempt to get more detailed error information from the driver
            if hasattr(self._driver, "format_connection_error_details"):
                detailed_error_info = self._driver.format_connection_error_details(e)
                if detailed_error_info:
                    log_msg += f"\n--- Driver Specific Error Details ---\n{detailed_error_info}"
                    log_msg += "\n-------------------------------------"
            self.logger.error(log_msg, exc_info=True)  # exc_info=True will add traceback
            return False

    async def disconnect(self) -> None:
        """Disconnect from the message broker."""
        await self._driver.disconnect()

    async def publish(
        self, topic: str, payload: BaseMessage, qos: int = 0, retain: bool = False
    ) -> None:
        """Publish a message to the specified topic.

        Args:
            topic: Topic to publish the message to
            payload: Message payload object
            qos: Quality of Service level (0, 1, or 2)
            retain: Whether the message should be retained by the broker
        """
        await self._driver.publish(topic, payload, qos=qos, retain=retain)

    async def subscribe(
        self, topic: str, handler: MessageHandler, message_cls: BaseMessage = BaseMessage, **kwargs
    ) -> None:
        """Subscribe to a topic with the specified message handler.

        Args:
            topic: Topic pattern to subscribe to
            handler: Callback function to handle received messages
            message_cls: Message class for parsing received payloads
            **kwargs: Additional arguments passed to the underlying driver
        """
        self.logger.info(
            f"Subscribing to topic: {topic} with handler: {handler.__name__}, "
            f"message_cls: {message_cls.__name__}"
        )

        # Make handler_adapter async and handle task creation properly
        async def handler_adapter(payload: bytes) -> None:
            try:
                # parse raw payload into BaseMessage
                base_msg = BaseMessage.model_validate_json(payload.decode("utf-8"))
            except Exception as e:
                self.logger.error(f"Failed to parse BaseMessage: {e}", exc_info=True)
                return

            if message_cls is not BaseMessage:
                # Check if the message is of the expected type
                try:
                    specific = message_cls.model_validate_json(payload.decode("utf-8"))
                    # Create and forget task - don't return it
                    asyncio.create_task(handler(specific))
                    return
                except ValidationError as e:
                    # extract error fields and ignore if only 'type' field is invalid
                    errors = e.errors()
                    fields = {err.get("loc", (None,))[0] for err in errors}
                    if fields != {"type"}:
                        self.logger.error(
                            f"Failed to parse message to {message_cls.__name__}: {e}", exc_info=True
                        )
                    return
                except Exception as e:
                    self.logger.error(
                        f"Failed to parse message to {message_cls.__name__}: {e}", exc_info=True
                    )
                    return

            # Create and forget task for generic subscription
            asyncio.create_task(handler(base_msg))
            # Don't return the task

        await self._driver.subscribe(topic, handler_adapter, **kwargs)

    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a previously subscribed topic.

        Args:
            topic: Topic to unsubscribe from
        """
        if hasattr(self._driver, "unsubscribe"):
            await self._driver.unsubscribe(topic)

    async def log(self, message: str, level: str = "info") -> None:
        """Publish a log message to the configured log topic.

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

            # publish uses driver internally
            await self.publish(self.pub_log_topic, log_payload)
        except Exception as e:
            self.logger.error(f"Failed to publish log message: {e}", exc_info=True)

    async def stop(self) -> None:
        """Stop the messager client, disconnecting from broker and cleaning up resources.

        This is an alias for disconnect() to provide a consistent interface.
        """
        await self.disconnect()
