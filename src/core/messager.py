import paho.mqtt.client as mqtt
import json
import time
import threading
import logging  # Import the logging module
from typing import Dict, Any, Optional, Callable, Union

from core.logger import get_logger, LogLevel, parse_log_level

# Define type aliases for clarity
MQTTClient = mqtt.Client
MQTTMessage = mqtt.MQTTMessage
MQTTMessageHandler = Callable[[MQTTClient, Any, MQTTMessage], None]  # Original handler signature
MessageHandlerWrapper = Callable[[MQTTMessage], None]  # Simplified wrapper signature

# Disable Paho MQTT DEBUG logging to prevent console spam
mqtt_logger = logging.getLogger("paho.mqtt.client")
mqtt_logger.setLevel(logging.WARNING)  # Only show warnings and above
mqtt_logger.propagate = False  # Prevent log propagation


class Messager:
    """Handles MQTT connection, publishing, and subscription logic."""

    def __init__(
        self,
        broker_address: str,
        port: int,
        client_id: str,
        keepalive: int,
        pub_log_topic: str,
        log_level: Union[str, LogLevel] = LogLevel.INFO,
    ):
        self.broker_address = broker_address
        self.port = port
        self.client_id = client_id
        self.keepalive = keepalive
        self.pub_log_topic = pub_log_topic

        # Set up logger
        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level

        self.logger = get_logger(f"mqtt.{self.client_id}", self.log_level)

        self.client: MQTTClient = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id
        )
        # Improve reliability: auto reconnect delays and higher inflight/queue limits
        self.client.reconnect_delay_set(min_delay=1, max_delay=120)
        # Allow larger message flows
        self.client.max_inflight_messages_set(100)
        self.client.max_queued_messages_set(0)  # unlimited
        # Assign callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

        self._topic_handlers: Dict[str, MessageHandlerWrapper] = {}
        self._is_connected = False
        self._connection_event = threading.Event()  # Event for waiting

        self.logger.debug("Messager instance initialized")

    def set_log_level(self, level: Union[str, LogLevel]) -> None:
        """
        Set the log level for this Messager instance

        Args:
            level: New log level (string or LogLevel enum)
        """
        if isinstance(level, str):
            self.log_level = parse_log_level(level)
        else:
            self.log_level = level

        self.logger.setLevel(self.log_level.value)
        self.logger.info(f"Log level changed to {self.log_level.name}")

        # Update handlers
        for handler in self.logger.handlers:
            handler.setLevel(self.log_level.value)

    def subscribe(self, topic: str, handler: MessageHandlerWrapper, qos: int = 1) -> None:
        """Subscribes to a topic with a handler."""
        self.logger.info(f"Registering subscription for topic: {topic} (qos={qos})")
        self._topic_handlers[topic] = handler
        # If already connected, perform the MQTT subscribe now
        if self._is_connected:
            self.logger.info(f"Subscribing to topic: {topic} (qos={qos})")
            self.client.subscribe(topic, qos=qos)

    def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic and remove its handler."""
        if not self.is_connected():
            self.logger.error(f"Cannot unsubscribe from {topic}: not connected")
            return

        try:
            # Unsubscribe from the topic
            result, _ = self.client.unsubscribe(topic)
            if result == mqtt.MQTT_ERR_SUCCESS:
                self.logger.debug(f"Unsubscribed from topic: {topic}")
                # Remove the handler after successful unsubscription
                self._topic_handlers.pop(topic, None)
            else:
                self.logger.error(
                    f"Failed to unsubscribe from topic {topic} (error code: {result})"
                )
        except Exception as e:
            self.logger.error(f"Error unsubscribing from topic {topic}: {e}")

    def _on_connect(
        self,
        client: MQTTClient,
        userdata: Any,
        flags: Dict[str, int],
        rc: int,
        properties=None,
    ) -> None:
        """Internal callback for MQTT connection."""
        if rc == 0:
            self._is_connected = True
            self._connection_event.set()  # Signal connection success
            self.logger.info(f"Connected successfully to MQTT Broker: {self.broker_address}")
            self.mqtt_log(f"Connected to MQTT Broker: {self.broker_address}")
            # Resubscribe to topics upon reconnection (iterate over a static copy of keys)
            for topic in list(self._topic_handlers.keys()):
                self.logger.info(f"Subscribing to topic: {topic} (qos=1)")
                # Subscribe with QoS 1 for reliable delivery
                self.client.subscribe(topic, qos=1)
        else:
            self._is_connected = False
            self._connection_event.clear()  # Ensure event is clear on failure
            self.logger.error(f"Failed to connect to MQTT Broker, return code {rc}")
            # Note: mqtt_log won't work here since we're not connected

    def _on_message(self, client: MQTTClient, userdata: Any, msg: MQTTMessage) -> None:
        """Internal callback for received messages."""
        topic = msg.topic
        # Log receipt of every message at debug level to avoid flooding logs
        self.logger.debug(f"Received raw message on topic {topic}, payload: {msg.payload}")
        # Also publish to log topic if connected
        try:
            if self._is_connected:
                self.mqtt_log(
                    f"Received message on topic '{topic}': {msg.payload.decode('utf-8', errors='ignore')}",
                    level="debug",
                )
        except Exception:
            pass

        # Find matching handler, support wildcard subscriptions
        handler = None
        for pattern, h in self._topic_handlers.items():
            if mqtt.topic_matches_sub(pattern, topic):
                handler = h
                break

        if handler:
            try:
                # Pass only the message to the simplified handler wrapper
                handler(msg)
            except Exception as e:
                err_msg = f"Unhandled exception in handler for topic '{topic}': {e}"
                self.logger.error(err_msg)
                try:
                    self.mqtt_log(err_msg, level="error")
                except Exception:  # Avoid errors if publish fails during handler error
                    self.logger.error("Failed to publish handler error log.")
        else:
            self.logger.warning(f"No handler registered for received message on topic: {topic}")

    def _on_disconnect(
        self,
        client: MQTTClient,
        userdata: Any,
        *args: Any,
    ) -> None:
        """Internal callback for MQTT disconnection (supports multiple callback signatures)."""
        self._is_connected = False
        self._connection_event.clear()  # Signal disconnection
        # Log disconnection (args may include rc or flags/reason/properties)
        self.logger.info("Disconnected from MQTT Broker")

    def _on_publish(
        self,
        client: MQTTClient,
        userdata: Any,
        mid: int,
        *args: Any,
    ) -> None:
        """Internal callback when a message is successfully published (optional, supports multiple signatures)."""
        # Can log publication mid at debug level if needed
        self.logger.debug(f"Message published (MID: {mid})")
        pass

    def register_handler(self, topic: str, handler: MessageHandlerWrapper) -> None:
        """Registers a handler function for a specific topic."""
        self.logger.info(f"Registering handler for topic: {topic}")
        self._topic_handlers[topic] = handler
        # If already connected, subscribe to the new topic immediately
        if self._is_connected:
            self.logger.info(f"Subscribing to newly registered topic: {topic} (qos=1)")
            self.client.subscribe(topic, qos=1)

    def connect(self, start_loop: bool = True) -> bool:  # Add start_loop parameter
        """Connects to the MQTT broker. Optionally starts background loop."""
        if self._is_connected:
            self.logger.info("Already connected.")
            return True
        try:
            self.logger.info(f"Connecting to MQTT Broker: {self.broker_address}:{self.port}...")
            self._connection_event.clear()
            self.client.connect(self.broker_address, self.port, self.keepalive)
            # Only start background loop if requested (e.g., for client)
            if start_loop:
                self.logger.info("Starting background loop in connect().")
                self.client.loop_start()
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to MQTT broker: {e}")
            self._is_connected = False
            if start_loop:
                try:
                    self.client.loop_stop()  # Stop if connect failed after loop_start
                except Exception:
                    pass  # Ignore errors stopping a potentially non-started loop
            return False

    def publish(
        self, topic: str, payload: Union[str, Dict[str, Any]], retain: bool = False
    ) -> bool:
        """Publishes a message to a given MQTT topic, queuing if broker is not connected."""
        try:
            if not isinstance(payload, str):
                payload = json.dumps(payload)
            # Log publication of message (at debug level to avoid flooding logs)
            self.logger.debug(f"Publishing to topic {topic}, payload: {payload}")
            # Publish with QoS 1 to ensure delivery; unlimited queuing is enabled
            self.client.publish(topic, payload, qos=1, retain=retain)
            return True
        except Exception as e:
            self.logger.error(f"Error publishing MQTT message to {topic}: {e}")
            return False

    def mqtt_log(self, message: str, level: str = "info") -> None:
        """Publishes a log message to the configured log topic."""
        log_payload = {"timestamp": time.time(), "level": level, "message": message}
        self.publish(self.pub_log_topic, log_payload)

    def log(self, message: str, level: str = "info") -> None:
        """Logs a message both to the internal logger and to MQTT if connected.

        Args:
            message: The log message
            level: Log level (debug, info, warning, error)
        """
        # Log to internal logger
        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        else:
            self.logger.info(message)  # Default to info

        # Log to MQTT as well
        try:
            self.mqtt_log(message, level)
        except Exception as e:
            self.logger.error(f"Error sending log to MQTT: {e}")

    def start(
        self,
        subscriptions: Dict[str, str],
        handlers: Dict[str, MessageHandlerWrapper],
        wait_for_connection: bool = False,
        timeout: float = 10.0,
    ) -> bool:
        """Registers handlers, connects, optionally waits, and runs the background loop."""
        # 1. Register Handlers
        self.logger.info("Registering handlers...")
        for topic, handler_name in subscriptions.items():
            handler_func = handlers.get(handler_name)
            if handler_func:
                self.register_handler(topic, handler_func)
            else:
                self.logger.warning(
                    f"Handler function '{handler_name}' "
                    f"for topic '{topic}' not found in provided handlers."
                )

        # 2. Connect and start background loop
        if not self.connect(start_loop=True):
            return False

        # 3. Wait for Connection (Optional)
        if wait_for_connection:
            self.logger.info(f"Waiting for connection (timeout: {timeout}s)...")
            if not self._connection_event.wait(timeout=timeout):
                self.logger.error("Connection timed out.")
                self.stop()
                return False
            self.logger.info("Connection established.")

        # Messager is now connected and background loop is running.
        return True

    def start_background_loop(self) -> None:
        """Starts the MQTT network loop in a background thread."""
        # This method is primarily for the CLI client now
        if not self.is_connected():
            # Attempt connection if not connected, starting the loop
            self.logger.info("Not connected, attempting connect and background loop...")
            if not self.connect(start_loop=True):  # Connect *and* start loop
                self.logger.error("Connection failed for background loop.")
                return
            # Wait briefly for connection
            if not self._connection_event.wait(timeout=5.0):
                self.logger.error("Connection timeout for background loop.")
                self.disconnect()
                return
        else:
            # If already connected, ensure loop is started
            self.logger.info("Already connected, ensuring background loop is running...")
            self.client.loop_start()  # Safe to call multiple times

    def stop_background_loop(self) -> None:
        """Stops the MQTT network loop running in the background thread."""
        self.logger.info("Stopping MQTT background network loop...")
        self.client.loop_stop()

    def disconnect(self) -> None:
        """Disconnects from the MQTT broker."""
        if self._is_connected:
            self.logger.info("Disconnecting...")
            self.client.disconnect()
            # _on_disconnect will set self._is_connected = False
        else:
            self.logger.info("Already disconnected.")

    def stop(self) -> None:
        """Stops the Messager gracefully (disconnects)."""
        self.logger.info("Stop requested.")
        self.disconnect()

    def is_connected(self) -> bool:
        """Returns the connection status."""
        return self._is_connected
