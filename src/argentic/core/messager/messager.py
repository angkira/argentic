import time
import asyncio
from typing import Dict, Optional, Union, Any, Type, List
import ssl

from pydantic import ValidationError

from argentic.core.messager.drivers import create_driver
from argentic.core.messager.drivers.base_definitions import MessageHandler

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

        self._tls_params: Optional[Dict[str, Any]] = None
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

        # Prepare driver config
        config_data = {
            "url": broker_address,
            "port": port,
            "user": username,
            "password": password,
            "client_id": self.client_id,
            "keepalive": keepalive,
            **driver_kwargs,
        }

        self._driver = create_driver(protocol, config_data)

        # ------------------------------------------------------------------
        # Internal registry of handlers per topic. This allows us to attach
        # *multiple* high-level handlers to the same MQTT topic while issuing
        # only ONE subscription to the underlying driver. Each incoming
        # payload is then dispatched to every registered (handler,
        # message_cls) pair.  This prevents the "second subscription
        # overwrites the first" broker behaviour from dropping messages.
        # ------------------------------------------------------------------
        self._topic_handlers: Dict[str, List[tuple[MessageHandler, type]]] = {}
        # Store the dispatcher adapter we created for each topic so we can
        # cleanly unsubscribe later if needed.
        self._topic_dispatchers: Dict[str, MessageHandler] = {}

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
        self,
        topic: str,
        handler: MessageHandler,
        message_cls: Type[BaseMessage] = BaseMessage,
        **kwargs,
    ) -> None:
        """Subscribe to a topic with the specified message handler.

        Args:
            topic: Topic pattern to subscribe to
            handler: Callback function to handle received messages
            message_cls: Message class for parsing received payloads
            **kwargs: Additional arguments passed to the underlying driver
        """

        self.logger.info(
            f"Registering handler {handler.__name__} (cls={message_cls.__name__}) for topic '{topic}'"
        )

        # Lazily create the dispatcher for this topic on first registration
        if topic not in self._topic_handlers:
            self._topic_handlers[topic] = []

            async def _dispatcher(payload: BaseMessage) -> None:
                """Dispatch an incoming BaseMessage to all registered
                (handler, message_cls) pairs for this topic."""

                # Create a snapshot of handlers to avoid mutation-during-iter
                for _handler, _cls in list(self._topic_handlers.get(topic, [])):
                    if _cls is not BaseMessage:
                        expected_type = None

                        # 1️⃣ Literal annotation check
                        if hasattr(_cls, "__annotations__"):
                            type_annotation = _cls.__annotations__.get("type")
                            if (
                                type_annotation
                                and hasattr(type_annotation, "__args__")
                                and type_annotation.__args__
                            ):
                                expected_type = type_annotation.__args__[0]

                        # 2️⃣ Default field value fallback
                        if expected_type is None:
                            model_fields = getattr(_cls, "model_fields", None)
                            if model_fields is not None:
                                type_field = (
                                    model_fields.get("type")
                                    if isinstance(model_fields, dict)
                                    else None
                                )
                                if type_field and hasattr(type_field, "default"):
                                    expected_type = type_field.default
                            else:
                                expected_type = getattr(_cls, "type", None)

                        if expected_type and payload.type != expected_type:
                            # Not the right message type for this handler – skip
                            continue

                        try:
                            original_json = getattr(payload, "_original_json", None)
                            validate_json = getattr(_cls, "model_validate_json", None)
                            if original_json and callable(validate_json):
                                specific_msg = validate_json(original_json)
                            else:
                                # Fallback to generic model_validate (runtime checked)
                                specific_msg = getattr(_cls, "model_validate")(payload.model_dump())

                            await _handler(specific_msg)  # type: ignore[arg-type]
                        except ValidationError:
                            # Silent skip if validation fails – another handler might match
                            continue
                        except Exception as exc:
                            self.logger.error(
                                f"Error in handler {_handler.__name__} for topic '{topic}': {exc}",
                                exc_info=True,
                            )
                    else:
                        # Generic BaseMessage handler – forward as-is
                        try:
                            await _handler(payload)
                        except Exception as exc:
                            self.logger.error(
                                f"Error in handler {_handler.__name__} for topic '{topic}': {exc}",
                                exc_info=True,
                            )

            # Save dispatcher so we can unsubscribe later
            self._topic_dispatchers[topic] = _dispatcher

            # Only the FIRST registration performs the low-level subscribe
            await self._driver.subscribe(topic, _dispatcher, **kwargs)

        # Always register the (handler, message_cls) pair in local registry
        self._topic_handlers[topic].append((handler, message_cls))

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
            log_payload = BaseMessage(
                type="log",
                source=self.client_id,
                data={
                    "timestamp": time.time(),
                    "level": level,
                    "message": message,
                },
            )

            # publish uses driver internally
            await self.publish(self.pub_log_topic, log_payload)
        except Exception as e:
            self.logger.error(f"Failed to publish log message: {e}", exc_info=True)

    async def stop(self) -> None:
        """Stop the messager client, disconnecting from broker and cleaning up resources.

        This is an alias for disconnect() to provide a consistent interface.
        """
        await self.disconnect()
