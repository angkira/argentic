import time
import asyncio  # for scheduling async message handlers
from typing import Dict, Optional, Union, Any
import ssl

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
        )

        self._driver = create_driver(protocol, cfg)

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

        # Handler adapter: parse to BaseMessage, filter by type, then parse to specific model
        def handler_adapter(payload: bytes) -> None:
            try:
                self.logger.info(
                    f"Parsing message to specific class: [{message_cls["type"]}] or {message_cls.__annotations__['type']}"
                )
                # parse raw payload into BaseMessage
                base_msg = BaseMessage.model_validate_json(payload.decode("utf-8"))
            except Exception as e:
                self.logger.error(f"Failed to parse BaseMessage: {e}", exc_info=True)
                return

            # self.logger.info(
            #     f"Expected class: {message_cls.__name__}; Parsed BaseMessage: {base_msg.model_dump_json()}, {message_cls is not BaseMessage}"
            # )

            if message_cls is not BaseMessage:

                # Check if the message is of the expected type
                try:
                    specific = message_cls.model_validate_json(payload.decode("utf-8"))
                    return asyncio.create_task(handler(specific))
                except ValidationError as e:
                    # extract error fields and ignore if only 'type' field is invalid
                    errors = e.errors()
                    fields = {err.get('loc', (None,))[0] for err in errors}
                    if fields != {'type'}:
                        self.logger.error(
                            f"Failed to parse message to {message_cls.__name__}: {e}", exc_info=True
                        )
                    return
                except Exception as e:
                    self.logger.error(
                        f"Failed to parse message to {message_cls.__name__}: {e}", exc_info=True
                    )
                    return
                # generic subscription
            return asyncio.create_task(handler(base_msg))

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
