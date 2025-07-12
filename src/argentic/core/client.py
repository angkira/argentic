import uuid
import asyncio
from typing import Optional

from argentic.core.messager.messager import Messager
from argentic.core.protocol.message import AskQuestionMessage, AnswerMessage, BaseMessage
from argentic.core.logger import get_logger, LogLevel


class Client:
    """Base class for clients that communicate with the agent via MQTT"""

    def __init__(
        self,
        messager: Messager,
        user_id: Optional[str] = None,
        client_id: Optional[str] = None,
        log_level: LogLevel = LogLevel.INFO,
    ):
        """
        Initialize a Client instance

        Args:
            messager: The Messager instance to use for communication
            user_id: Unique identifier for the user (defaults to UUID if not provided)
            client_id: Identifier for this client instance (defaults to class name + UUID)
            log_level: The logging level for this client
        """
        self.messager = messager
        self.user_id = user_id or str(uuid.uuid4())
        self.client_id = client_id or f"{self.__class__.__name__}_{uuid.uuid4()}"
        self.logger = get_logger(f"client.{self.__class__.__name__}", log_level)

        # Track subscribed topics
        self._subscribed_topics = set()

        # Configure standard topics
        self.answer_topic = None  # Will be set during registration

    async def connect(self) -> bool:
        """Connect to the MQTT broker. Returns True if successful."""
        self.logger.info(f"Connecting to MQTT broker as {self.client_id}...")

        try:
            connect_result = await self.messager.connect()
            if not connect_result:
                self.logger.error("Connection failed or timed out")
                return False

            self.logger.info("Connected to MQTT broker")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to MQTT: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        self.logger.info("Disconnecting from MQTT broker")
        try:
            await self.messager.stop()
        except Exception as e:
            self.logger.error(f"Error disconnecting: {e}")

    async def register_handlers(self, answer_topic: str) -> None:
        """
        Register standard message handlers

        Args:
            answer_topic: Topic to subscribe for answers
        """
        self.logger.info(f"Subscribing to answer topic: {answer_topic}")
        self.answer_topic = answer_topic

        # Subscribe to answer messages with the Pydantic model for type safety
        await self.messager.subscribe(answer_topic, self.handle_answer, AnswerMessage)
        self._subscribed_topics.add(answer_topic)

    async def handle_answer(self, message: AnswerMessage) -> None:
        """
        Handle answer messages - override in subclasses

        Args:
            message: The answer message to handle
        """
        self.logger.info(f"Received answer: {message.question}")
        # Base implementation - subclasses should override this

    async def ask_question(self, question: str, topic: str) -> None:
        """
        Ask a question to the agent

        Args:
            question: The question text
            topic: Topic to publish the question to
        """
        self.logger.info(f"Asking question: {question}")
        ask_message = AskQuestionMessage(
            question=question,
            user_id=self.user_id,
            source=self.client_id,
        )
        await self.messager.publish(topic, ask_message)

    async def start(self) -> bool:
        """Start the client - connect"""
        return await self.connect()

    async def stop(self) -> None:
        """Stop the client - unsubscribe and disconnect"""
        # Unsubscribe from all topics
        for topic in self._subscribed_topics:
            try:
                await self.messager.unsubscribe(topic)
            except Exception as e:
                self.logger.error(f"Error unsubscribing from {topic}: {e}")

        # Disconnect from broker
        await self.disconnect()
