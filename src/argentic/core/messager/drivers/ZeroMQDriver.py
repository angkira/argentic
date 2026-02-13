"""ZeroMQ driver for high-performance brokerless messaging.

This driver implements pub/sub messaging using ZeroMQ's XPUB/XSUB proxy pattern.
Unlike traditional brokers, it provides ultra-low latency (~50-100μs) but without
message persistence or QoS guarantees.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Type

from argentic.core.messager.drivers.base_definitions import BaseDriver, MessageHandler
from argentic.core.protocol.message import BaseMessage

from .configs import ZeroMQDriverConfig
from .zmq_proxy import ZMQProxyManager

try:
    import zmq
    import zmq.asyncio

    ZMQ_INSTALLED = True
except ImportError:
    ZMQ_INSTALLED = False
    # Define dummy types for type hinting
    zmq = type("zmq", (object,), {"asyncio": type("asyncio", (object,), {"Context": object})})

logger = logging.getLogger(__name__)


class ZeroMQDriver(BaseDriver[ZeroMQDriverConfig]):
    """ZeroMQ driver with XPUB/XSUB proxy support.

    Architecture:
    - PUB socket: Connects to proxy frontend (XSUB) for publishing
    - SUB socket: Connects to proxy backend (XPUB) for subscribing
    - Proxy: Optional embedded or external XPUB/XSUB router

    Wire format: "<topic> <json_payload>"
    Example: "agent/command/ask_question {\"type\":\"ask_question\",...}"
    """

    def __init__(self, config: ZeroMQDriverConfig):
        """Initialize ZeroMQ driver.

        Args:
            config: ZeroMQ driver configuration

        Raises:
            ImportError: If pyzmq is not installed
        """
        if not ZMQ_INSTALLED:
            raise ImportError(
                "pyzmq is not installed. "
                "Please install it with: pip install argentic[zeromq]"
            )
        super().__init__(config)

        # ZeroMQ context and sockets
        self._context: Optional[zmq.asyncio.Context] = None
        self._pub_socket: Optional[zmq.asyncio.Socket] = None
        self._sub_socket: Optional[zmq.asyncio.Socket] = None

        # Proxy manager (if embedded mode)
        self._proxy_manager: Optional[ZMQProxyManager] = None

        # Message handling
        self._listeners: Dict[str, List[MessageHandler]] = {}
        self._reader_task: Optional[asyncio.Task] = None

        # Connection state
        self._connected = False

    async def connect(self) -> bool:
        """Connect to ZeroMQ proxy and initialize sockets.

        Workflow:
        1. Start proxy if embedded mode
        2. Create ZeroMQ context
        3. Create and configure PUB/SUB sockets
        4. Connect sockets to proxy
        5. Start reader task

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        if self._connected:
            logger.debug("Already connected to ZeroMQ")
            return True

        try:
            # Start embedded proxy if configured
            if self.config.start_proxy and self.config.proxy_mode == "embedded":
                frontend_url = f"tcp://{self.config.url}:{self.config.port}"
                backend_url = f"tcp://{self.config.url}:{self.config.backend_port}"

                self._proxy_manager = ZMQProxyManager(frontend_url, backend_url)
                self._proxy_manager.start()
                logger.info("Embedded ZeroMQ proxy started")

                # Wait for proxy to be ready
                await asyncio.sleep(0.2)

            # Create context and sockets
            self._context = zmq.asyncio.Context()

            # PUB socket for publishing (connects to frontend)
            self._pub_socket = self._context.socket(zmq.PUB)
            self._pub_socket.setsockopt(zmq.SNDHWM, self.config.high_water_mark)
            self._pub_socket.setsockopt(zmq.LINGER, self.config.linger)

            pub_url = f"tcp://{self.config.url}:{self.config.port}"
            self._pub_socket.connect(pub_url)
            logger.debug(f"PUB socket connected to {pub_url}")

            # SUB socket for subscribing (connects to backend)
            self._sub_socket = self._context.socket(zmq.SUB)
            self._sub_socket.setsockopt(zmq.RCVHWM, self.config.high_water_mark)

            sub_url = f"tcp://{self.config.url}:{self.config.backend_port}"
            self._sub_socket.connect(sub_url)
            logger.debug(f"SUB socket connected to {sub_url}")

            # Start reader task
            self._reader_task = asyncio.create_task(self._reader())
            logger.debug("Reader task started")

            self._connected = True
            logger.info("ZeroMQ driver connected successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to ZeroMQ: {e}", exc_info=True)
            await self.disconnect()
            raise ConnectionError(f"ZeroMQ connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from ZeroMQ and cleanup resources.

        Workflow:
        1. Stop reader task
        2. Close sockets
        3. Terminate context
        4. Stop proxy if embedded
        5. Clear state
        """
        if not self._connected:
            logger.debug("Not connected, skipping disconnect")
            return

        logger.info("Disconnecting ZeroMQ driver")

        # Stop reader task
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                logger.debug("Reader task cancelled")

        # Close sockets
        if self._pub_socket:
            self._pub_socket.close(linger=0)
            self._pub_socket = None

        if self._sub_socket:
            self._sub_socket.close(linger=0)
            self._sub_socket = None

        # Terminate context
        if self._context:
            self._context.term()
            self._context = None

        # Stop embedded proxy
        if self._proxy_manager:
            self._proxy_manager.stop()
            self._proxy_manager = None

        # Clear state
        self._listeners.clear()
        self._reader_task = None
        self._connected = False

        logger.info("ZeroMQ driver disconnected")

    async def publish(
        self, topic: str, payload: BaseMessage, qos: int = 0, retain: bool = False
    ) -> None:
        """Publish message to topic.

        Wire format: "<topic> <json_payload>"

        Note: qos and retain parameters are ignored (ZeroMQ doesn't support them).

        Args:
            topic: Topic to publish to (must not contain spaces)
            payload: Message to publish
            qos: Ignored (for API compatibility)
            retain: Ignored (for API compatibility)

        Raises:
            ConnectionError: If not connected
            ValueError: If topic contains spaces
        """
        if not self._connected or not self._pub_socket:
            raise ConnectionError("ZeroMQ driver not connected")

        if " " in topic:
            raise ValueError(
                f"Topic '{topic}' contains spaces, which are not allowed in ZeroMQ driver. "
                "Use slashes or underscores instead."
            )

        # Serialize message
        json_data = payload.model_dump_json()

        # Format: "<topic> <json>"
        message = f"{topic} {json_data}"

        # Send with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self._pub_socket.send_string(message, flags=zmq.NOBLOCK)
                logger.debug(f"Published to {topic}: {len(json_data)} bytes")
                return
            except zmq.Again:
                # Send buffer full, wait and retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.01 * (attempt + 1))
                else:
                    raise ConnectionError(
                        f"Failed to publish to {topic}: send buffer full after {max_retries} retries"
                    )
            except zmq.ZMQError as e:
                logger.error(f"ZMQ error publishing to {topic}: {e}")
                raise ConnectionError(f"ZMQ publish error: {e}") from e

    async def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        message_cls: Type[BaseMessage] = BaseMessage,
        **kwargs,
    ) -> None:
        """Subscribe to topic and register handler.

        ZeroMQ uses prefix matching for topics. Subscribing to "agent/command"
        will receive messages for "agent/command/ask_question", etc.

        Args:
            topic: Topic to subscribe to
            handler: Async callback for messages
            message_cls: Message class for validation (currently unused)
            **kwargs: Additional arguments (ignored)

        Raises:
            ConnectionError: If not connected
        """
        if not self._connected or not self._sub_socket:
            raise ConnectionError("ZeroMQ driver not connected")

        # Register handler
        if topic not in self._listeners:
            self._listeners[topic] = []

            # Subscribe socket to topic (prefix matching)
            topic_bytes = topic.encode(self.config.topic_encoding)
            self._sub_socket.setsockopt(zmq.SUBSCRIBE, topic_bytes)
            logger.debug(f"Subscribed to topic: {topic}")

        self._listeners[topic].append(handler)
        logger.debug(f"Handler registered for {topic} ({len(self._listeners[topic])} total)")

    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from topic and remove all handlers.

        Args:
            topic: Topic to unsubscribe from
        """
        if topic in self._listeners:
            # Unsubscribe socket
            if self._sub_socket:
                topic_bytes = topic.encode(self.config.topic_encoding)
                self._sub_socket.setsockopt(zmq.UNSUBSCRIBE, topic_bytes)
                logger.debug(f"Unsubscribed from topic: {topic}")

            # Remove handlers
            del self._listeners[topic]

    async def _reader(self) -> None:
        """Reader loop for incoming messages.

        Continuously receives messages from SUB socket, parses topic and payload,
        and dispatches to registered handlers. Runs until cancelled.

        Message format: "<topic> <json_payload>"
        """
        logger.debug("Reader loop started")

        while True:
            try:
                if not self._sub_socket:
                    logger.warning("SUB socket not available, stopping reader")
                    break

                # Receive message (blocks until available)
                message = await self._sub_socket.recv_string()

                # Parse: "<topic> <json>"
                parts = message.split(" ", 1)
                if len(parts) != 2:
                    logger.warning(f"Invalid message format (expected '<topic> <json>'): {message[:100]}")
                    continue

                topic, json_data = parts

                # Deserialize message
                try:
                    msg_obj = BaseMessage.model_validate_json(json_data)
                except Exception as e:
                    logger.warning(f"Failed to deserialize message from {topic}: {e}")
                    continue

                # Dispatch to handlers
                # Find all matching topics (prefix matching)
                matched_handlers = []
                for registered_topic, handlers in self._listeners.items():
                    if topic.startswith(registered_topic):
                        matched_handlers.extend(handlers)

                if not matched_handlers:
                    logger.debug(f"No handlers for topic: {topic}")
                    continue

                # Call handlers
                for handler in matched_handlers:
                    try:
                        await handler(msg_obj)
                    except Exception as e:
                        logger.error(
                            f"Handler error for topic {topic}: {e}",
                            exc_info=True
                        )

            except asyncio.CancelledError:
                logger.debug("Reader loop cancelled")
                break
            except zmq.ZMQError as e:
                logger.error(f"ZMQ error in reader: {e}")
                # Try to reconnect
                await self._reconnect()
            except Exception as e:
                logger.error(f"Unexpected error in reader: {e}", exc_info=True)
                await asyncio.sleep(0.1)  # Avoid tight loop on persistent errors

        logger.debug("Reader loop stopped")

    async def _reconnect(self) -> None:
        """Attempt to reconnect sockets and resubscribe to topics.

        This is called when a ZMQ error occurs in the reader loop.
        """
        logger.warning("Attempting to reconnect ZeroMQ sockets")

        try:
            # Save current subscriptions
            subscriptions = list(self._listeners.keys())

            # Close and recreate sockets
            if self._pub_socket:
                self._pub_socket.close(linger=0)

            if self._sub_socket:
                self._sub_socket.close(linger=0)

            # Recreate sockets
            if self._context:
                # PUB socket
                self._pub_socket = self._context.socket(zmq.PUB)
                self._pub_socket.setsockopt(zmq.SNDHWM, self.config.high_water_mark)
                self._pub_socket.setsockopt(zmq.LINGER, self.config.linger)
                pub_url = f"tcp://{self.config.url}:{self.config.port}"
                self._pub_socket.connect(pub_url)

                # SUB socket
                self._sub_socket = self._context.socket(zmq.SUB)
                self._sub_socket.setsockopt(zmq.RCVHWM, self.config.high_water_mark)
                sub_url = f"tcp://{self.config.url}:{self.config.backend_port}"
                self._sub_socket.connect(sub_url)

                # Resubscribe to all topics
                for topic in subscriptions:
                    topic_bytes = topic.encode(self.config.topic_encoding)
                    self._sub_socket.setsockopt(zmq.SUBSCRIBE, topic_bytes)

                logger.info("ZeroMQ sockets reconnected successfully")

        except Exception as e:
            logger.error(f"Reconnection failed: {e}", exc_info=True)
            self._connected = False

    def is_connected(self) -> bool:
        """Check if driver is connected.

        Returns:
            True if connected and sockets are available
        """
        return self._connected and self._pub_socket is not None and self._sub_socket is not None

    def format_connection_error_details(self, error: Exception) -> Optional[str]:
        """Format ZeroMQ-specific connection error details.

        Args:
            error: Exception that occurred

        Returns:
            Formatted error message or None
        """
        if "zmq" in str(type(error)).lower():
            return f"ZeroMQ error: {error}"
        return None
