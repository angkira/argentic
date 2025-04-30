import paho.mqtt.client as mqtt
import json
import time
import threading
import logging  # Import the logging module
from typing import Dict, Any, Optional, Callable, Union

# Define type aliases for clarity
MQTTClient = mqtt.Client
MQTTMessage = mqtt.MQTTMessage
MQTTMessageHandler = Callable[
    [MQTTClient, Any, MQTTMessage], None
]  # Original handler signature
MessageHandlerWrapper = Callable[[MQTTMessage], None]  # Simplified wrapper signature

# --- Enable Paho MQTT Client Logging ---
# You can adjust the level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# DEBUG is very verbose and shows packet details.
mqtt_logger = logging.getLogger("paho.mqtt.client")
mqtt_logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed info
# You might want to configure a handler if you want logs to go to a file
# For now, let's just ensure the level is set. If you have a root logger
# configured elsewhere, Paho logs might appear there.
# If logs don't appear, add a basic handler:
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# mqtt_logger.addHandler(handler)
# mqtt_logger.propagate = False # Prevent duplicate logs if root logger exists


class Messager:
    """Handles MQTT connection, publishing, and subscription logic."""

    def __init__(
        self,
        broker_address: str,
        port: int,
        client_id: str,
        keepalive: int,
        pub_log_topic: str,
    ):
        self.broker_address = broker_address
        self.port = port
        self.client_id = client_id
        self.keepalive = keepalive
        self.pub_log_topic = pub_log_topic

        self.client: MQTTClient = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id
        )
        # --- Assign Paho Logger ---
        self.client.enable_logger(mqtt_logger)  # Add this line
        # --------------------------
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

        self._topic_handlers: Dict[str, MessageHandlerWrapper] = {}
        self._is_connected = False
        self._connection_event = threading.Event()  # Event for waiting

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
            print(
                f"Messager: Connected successfully to MQTT Broker: {self.broker_address}"
            )
            self.publish_log(f"Connected to MQTT Broker: {self.broker_address}")
            # Resubscribe to topics upon reconnection
            for topic in self._topic_handlers.keys():
                print(f"Messager: Subscribing to topic: {topic}")
                self.client.subscribe(topic)
        else:
            self._is_connected = False
            self._connection_event.clear()  # Ensure event is clear on failure
            print(f"Messager: Failed to connect to MQTT Broker, return code {rc}")
            # Note: publish_log won't work here

    def _on_message(self, client: MQTTClient, userdata: Any, msg: MQTTMessage) -> None:
        """Internal callback for received messages."""
        topic = msg.topic
        print(f"Messager: Received raw message on topic {topic}")  # Debugging
        handler = self._topic_handlers.get(topic)
        if handler:
            try:
                # Pass only the message to the simplified handler wrapper
                handler(msg)
            except Exception as e:
                err_msg = (
                    f"Messager: Unhandled exception in handler for topic '{topic}': {e}"
                )
                print(err_msg)
                try:
                    self.publish_log(err_msg, level="error")
                except Exception:  # Avoid errors if publish fails during handler error
                    print("Messager: Failed to publish handler error log.")
        else:
            print(
                f"Messager: Warning: No handler registered for received message on topic: {topic}"
            )

    def _on_disconnect(
        self,
        client: MQTTClient,
        userdata: Any,
        disconnect_flags: mqtt.DisconnectFlags,  # Added flags parameter
        reason_code: mqtt.ReasonCode,  # Use reason_code explicitly
        properties: Optional[mqtt.Properties] = None,  # Keep properties optional
    ) -> None:
        """Internal callback for MQTT disconnection."""
        self._is_connected = False
        self._connection_event.clear()  # Signal disconnection
        # Use reason_code in the log message
        log_msg = f"Messager: Disconnected from MQTT Broker (Reason: {reason_code})."
        print(log_msg)
        # Avoid trying to publish log if the disconnection was unexpected (rc != 0)
        # or if publish fails. Consider adding reconnection logic here if needed.

    def _on_publish(
        self,
        client: MQTTClient,
        userdata: Any,
        mid: int,
        properties=None,
        reasoncode=None,
    ) -> None:
        """Internal callback when a message is successfully published (optional)."""
        # print(f"Messager: Message Published (MID: {mid})") # Can be verbose
        pass

    def register_handler(self, topic: str, handler: MessageHandlerWrapper) -> None:
        """Registers a handler function for a specific topic."""
        print(f"Messager: Registering handler for topic: {topic}")
        self._topic_handlers[topic] = handler
        # If already connected, subscribe to the new topic immediately
        if self._is_connected:
            print(f"Messager: Subscribing to newly registered topic: {topic}")
            self.client.subscribe(topic)

    def connect(self, start_loop: bool = True) -> bool:  # Add start_loop parameter
        """Connects to the MQTT broker. Optionally starts background loop."""
        if self._is_connected:
            print("Messager: Already connected.")
            return True
        try:
            print(
                f"Messager: Connecting to MQTT Broker: {self.broker_address}:{self.port}..."
            )
            self._connection_event.clear()
            self.client.connect(self.broker_address, self.port, self.keepalive)
            # Only start background loop if requested (e.g., for client)
            if start_loop:
                print("Messager: Starting background loop in connect().")
                self.client.loop_start()
            return True
        except Exception as e:
            print(f"Messager: Error connecting to MQTT broker: {e}")
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
        """Publishes a message to a given MQTT topic."""
        if not self._is_connected:
            print(f"Messager: Not connected. Cannot publish to {topic}.")
            return False
        try:
            if not isinstance(payload, str):
                payload = json.dumps(payload)
            self.client.publish(topic, payload, retain=retain)
            # print(f"Messager: Published to {topic}: {payload[:100]}...") # Optional verbose log
            return True
        except Exception as e:
            print(f"Messager: Error publishing MQTT message to {topic}: {e}")
            return False

    def publish_log(self, message: str, level: str = "info") -> None:
        """Publishes a log message to the configured log topic."""
        log_payload = {"timestamp": time.time(), "level": level, "message": message}
        self.publish(self.pub_log_topic, log_payload)

    def start(
        self,
        subscriptions: Dict[str, str],
        handlers: Dict[str, MessageHandlerWrapper],
        wait_for_connection: bool = False,
        timeout: float = 10.0,
    ) -> bool:
        """Registers handlers, connects, optionally waits, and runs the background loop."""
        # 1. Register Handlers
        print("Messager: Registering handlers...")
        for topic, handler_name in subscriptions.items():
            handler_func = handlers.get(handler_name)
            if handler_func:
                self.register_handler(topic, handler_func)
            else:
                print(
                    f"Messager: Warning: Handler function '{handler_name}' "
                    f"for topic '{topic}' not found in provided handlers."
                )

        # 2. Connect and start background loop
        if not self.connect(start_loop=True):
            return False

        # 3. Wait for Connection (Optional)
        if wait_for_connection:
            print(f"Messager: Waiting for connection (timeout: {timeout}s)...")
            if not self._connection_event.wait(timeout=timeout):
                print("Messager: Connection timed out.")
                self.stop()
                return False
            print("Messager: Connection established.")

        # 4. Keep the process running until interruption
        print("Messager: Running background loop. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Messager: Loop interrupted by user (Ctrl+C). Shutting down.")
            self.publish_log("Service stopped by user.", level="info")
        except Exception as e:
            print(f"Messager: An error occurred in the main loop: {e}")
            try:
                self.publish_log(f"Main loop error: {e}", level="critical")
            except Exception:
                print("Messager: Failed to publish main loop error log.")
        finally:
            print("Messager: Stopping background loop and disconnecting.")
            self.stop()
        return True

    def start_background_loop(self) -> None:
        """Starts the MQTT network loop in a background thread."""
        # This method is primarily for the CLI client now
        if not self.is_connected():
            # Attempt connection if not connected, starting the loop
            print("Messager: Not connected, attempting connect and background loop...")
            if not self.connect(start_loop=True):  # Connect *and* start loop
                print("Messager: Connection failed for background loop.")
                return
            # Wait briefly for connection
            if not self._connection_event.wait(timeout=5.0):
                print("Messager: Connection timeout for background loop.")
                self.disconnect()
                return
        else:
            # If already connected, ensure loop is started
            print("Messager: Already connected, ensuring background loop is running...")
            self.client.loop_start()  # Safe to call multiple times

    def stop_background_loop(self) -> None:
        """Stops the MQTT network loop running in the background thread."""
        print("Messager: Stopping MQTT background network loop...")
        self.client.loop_stop()

    def disconnect(self) -> None:
        """Disconnects from the MQTT broker."""
        if self._is_connected:
            print("Messager: Disconnecting...")
            self.client.disconnect()
            # _on_disconnect will set self._is_connected = False
        else:
            print("Messager: Already disconnected.")

    def stop(self) -> None:
        """Stops the Messager gracefully (disconnects)."""
        print("Messager: Stop requested.")
        self.disconnect()

    def is_connected(self) -> bool:
        """Returns the connection status."""
        return self._is_connected
