import uuid
from typing import Dict, Any, Callable, Optional
import logging

from core.messager import Messager, MQTTMessage
from core.decorators import mqtt_handler_decorator
from core.protocol.message import AskQuestionMessage, AnswerMessage, from_mqtt_message, AnyMessage
from core.logger import get_logger, LogLevel


class Client:
    """Base class for clients that communicate with the agent via MQTT"""

    def __init__(
        self,
        messager: Messager,
        user_id: Optional[str] = None,
        client_id: Optional[str] = None,
        subscriptions: Optional[Dict[str, str]] = None,
        handlers: Optional[Dict[str, Callable]] = None,
        log_level: LogLevel = LogLevel.INFO,
    ):
        """
        Initialize a Client instance

        Args:
            messager: The Messager instance to use for communication
            user_id: Unique identifier for the user (defaults to UUID if not provided)
            client_id: Identifier for this client instance (defaults to class name + UUID)
            subscriptions: Dict mapping topics to handler names
            handlers: Dict mapping handler names to handler functions
            log_level: The logging level for this client
        """
        self.messager = messager
        self.user_id = user_id or str(uuid.uuid4())
        self.client_id = client_id or f"{self.__class__.__name__}_{uuid.uuid4()}"
        self._subscriptions = subscriptions or {}
        self._handlers = handlers or {}
        self.logger = get_logger(f"client.{self.__class__.__name__}", log_level)

    def connect(self) -> bool:
        """
        Connect to the MQTT broker

        Returns:
            bool: True if connection successful or already connected
        """
        self.logger.info(f"Connecting to MQTT broker as {self.client_id}...")
        if self.messager.is_connected():
            self.logger.info("Already connected to MQTT broker")
            return True

        connected = self.messager.connect()
        if not connected:
            self.logger.error("Failed to connect to MQTT broker")
            return False

        # Wait for connection confirmation
        if not self.messager._connection_event.wait(timeout=10.0):
            self.logger.error("Connection timeout")
            self.messager.disconnect()
            return False

        self.logger.info(
            f"Connected to MQTT broker at {self.messager.broker_address}:{self.messager.port}"
        )
        return True

    def disconnect(self) -> None:
        """Disconnect from the MQTT broker"""
        self.logger.info("Disconnecting from MQTT broker")
        self.messager.stop()

    def register_handlers(self) -> None:
        """Register all handlers for subscribed topics"""
        self.logger.info("Registering handlers...")
        for topic, handler_name in self._subscriptions.items():
            raw_handler_func = self._handlers.get(handler_name)
            if raw_handler_func:
                # Apply the decorator manually here
                decorated_handler = mqtt_handler_decorator(messager=self.messager)(raw_handler_func)
                self.messager.register_handler(topic, decorated_handler)
                self.logger.info(f"Registered handler for topic: {topic}")
            else:
                self.logger.warning(f"Handler function '{handler_name}' not found")

    def start(self) -> bool:
        """
        Start the client - connect and register handlers

        Returns:
            bool: True if startup successful
        """
        if not self.connect():
            return False

        self.register_handlers()
        self.messager.start_background_loop()
        return True

    def stop(self) -> None:
        """Stop the client - disconnect from MQTT broker"""
        self.disconnect()

    def ask_question(self, question: str, topic: str) -> None:
        """
        Ask a question to the agent

        Args:
            question: The question text
            topic: The topic to publish the question to
        """
        self.logger.info(f"Asking question: {question}")
        ask_message = AskQuestionMessage(
            question=question,
            user_id=self.user_id,
            source=self.client_id,
        )
        self.messager.publish(topic, ask_message.model_dump_json())
