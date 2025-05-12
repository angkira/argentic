import time
import asyncio  # for scheduling async message handlers
from typing import Dict, Optional, Union, Any
import ssl
import uuid

from pydantic import ValidationError

from core.messager.drivers import create_driver, DriverConfig, MessageHandler

from core.logger import get_logger, LogLevel, parse_log_level
from core.messager.protocols import MessagerProtocol
from core.protocol.message import BaseMessage


class Messager:
    """Async MQTT client"""

    def __init__(
        self,
        broker_address: str,
        port: int = 1883,
        protocol: MessagerProtocol = MessagerProtocol.MQTT,
        client_id: str = "",
        username: Optional[str] = None,
        password: Optional[str] = None,
        keepalive: int = 60,
        pub_log_topic: Optional[str] = None, # Log topic is now separate
        log_level: Union[LogLevel, str] = LogLevel.INFO,
        tls_params: Optional[Dict[str, Any]] = None,
        # Removed config: Dict[str, Any]
    ):
        # Restore original assignment logic
        self.broker_address = broker_address
        self.port = port
        self.client_id = client_id or f"client-{uuid.uuid4().hex[:8]}"
        self.username = username
        self.password = password
        self.keepalive = keepalive
        self.pub_log_topic = pub_log_topic # Assign passed log topic

        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level

        # Use client_id in logger name for clarity
        self.logger = get_logger(f"messager.{self.client_id}", level=self.log_level)

        # Restore TLS handling from passed tls_params
        self._tls_params = None
        if tls_params and isinstance(tls_params, dict):
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
                self._tls_params = {k: v for k, v in self._tls_params.items() if v is not None}
                if self._tls_params:
                     self.logger.info("TLS parameters configured.")
                else:
                     self._tls_params = None
            except Exception as e:
                self.logger.error(f"Failed to configure TLS parameters: {e}", exc_info=True)
                self._tls_params = None

        # Instantiate protocol driver using passed/default protocol
        cfg = DriverConfig(
            url=self.broker_address,
            port=self.port,
            user=self.username,
            password=self.password,
            token=None,
            client_id=self.client_id,
            keepalive=self.keepalive,
            tls_params=self._tls_params
        )

        # Use the protocol passed to init
        self._driver = create_driver(protocol, cfg)

        # Remove topic extraction logic from here - it belongs where Messager is instantiated
        # self.messaging_config = config.get("messaging", {})
        # topics_config = self.messaging_config.get("topics", {})
        # ... (removed topic assignments)
        # self.subscription_map = topics_config.get("subscriptions", {})

    def is_connected(self) -> bool:
        """Check if the client is currently connected."""
        return self._driver.is_connected()

    async def connect(self) -> bool:
        """Delegates connection to underlying driver"""
        try:
            await self._driver.connect()
            self.logger.info("Connected successfully via driver")
            return True
        except Exception as e:
            self.logger.error(f"Driver connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Delegates disconnection to underlying driver"""
        await self._driver.disconnect()

    async def publish(
        self, topic: str, payload: BaseMessage, qos: int = 0, retain: bool = False
    ) -> None:
        """Publishes a message via driver"""
        await self._driver.publish(topic, payload, qos=qos, retain=retain)

    async def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        message_cls: BaseMessage = BaseMessage,
    ) -> None:
        """Delegate subscription to driver"""

        self.logger.info(
            f"Subscribing to topic: {topic} with handler: {handler.__name__}, message_cls: {message_cls.__name__}"
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

        await self._driver.subscribe(topic, handler_adapter)

    async def unsubscribe(self, topic: str) -> None:
        """Delegate unsubscribe to driver if supported"""
        if hasattr(self._driver, "unsubscribe"):
            await self._driver.unsubscribe(topic)

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

            # publish uses driver internally
            await self.publish(self.pub_log_topic, log_payload)
        except Exception as e:
            self.logger.error(f"Failed to publish log message: {e}", exc_info=True)

    async def stop(self) -> None:
        """
        Stops the messager - disconnects from broker and cleans up resources.
        This is an alias for disconnect() to provide a consistent interface.
        """
        await self.disconnect()

    async def _connect_mqtt(self) -> None:
        """Establishes connection to the MQTT broker."""
        # Implementation of _connect_mqtt method
