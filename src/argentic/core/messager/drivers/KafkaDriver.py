from typing import Dict, List, Optional
from argentic.core.messager.drivers import BaseDriver, DriverConfig, MessageHandler
from argentic.core.protocol.message import BaseMessage
import asyncio  # Ensure asyncio is imported for Event

# Attempt to import aiokafka. If it fails, an ImportError will be raised,
# and this module should not be used for Kafka operations.
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from aiokafka.errors import KafkaConnectionError, KafkaTimeoutError
from aiokafka.abc import ConsumerRebalanceListener

# No AIOKAFKA_INSTALLED flag or dummy types needed anymore.
# If the imports above fail, this module won't load correctly for Kafka.

import logging
import uuid

# Get a logger instance
logger = logging.getLogger("KafkaDriver")


# Define a consumer rebalance listener to signal readiness
class _KafkaReadyListener(ConsumerRebalanceListener):
    def __init__(self, ready_event: asyncio.Event, topic_name: str):
        self._ready_event = ready_event
        self._topic_name = topic_name  # Store the target topic name
        super().__init__()

    async def on_partitions_assigned(self, assigned: List[TopicPartition]) -> None:
        logger.info(
            f"Partitions assigned for listener (target topic: {self._topic_name}): {assigned}"
        )
        # Check if any of the assigned partitions belong to our target topic
        relevant_partitions_assigned = any(tp.topic == self._topic_name for tp in assigned)

        if relevant_partitions_assigned:
            logger.info(
                f"Relevant partitions for topic '{self._topic_name}' assigned. Signaling consumer ready."
            )
            if (
                not self._ready_event.is_set()
            ):  # Check before setting to avoid issues if called multiple times
                self._ready_event.set()
        else:
            logger.info(
                f"No relevant partitions for topic '{self._topic_name}' in assignment: {assigned}. Not signaling ready yet."
            )

    async def on_partitions_revoked(self, revoked: List[TopicPartition]) -> None:
        logger.info(f"Partitions revoked: {revoked}")
        pass


class KafkaDriver(BaseDriver):
    def __init__(self, config: DriverConfig):
        # The imports at the top of the module handle the check.
        # If aiokafka wasn't found, an ImportError would have already occurred.
        super().__init__(config)
        self._producer: Optional[AIOKafkaProducer] = None
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._listeners: Dict[str, List[MessageHandler]] = {}
        self._reader_task: Optional[asyncio.Task] = None
        logger.debug(
            f"KafkaDriver initialized with config: {{url: {config.url}, port: {config.port}}}"
        )

    async def connect(self) -> None:
        servers = f"{self.config.url}:{self.config.port}"
        logger.info(f"Connecting Kafka producer to bootstrap servers: {servers}")
        try:
            # Explicitly pass the running event loop
            loop = asyncio.get_running_loop()
            self._producer = AIOKafkaProducer(bootstrap_servers=servers, loop=loop)
            await self._producer.start()
            logger.info("Kafka producer started successfully.")
        except KafkaConnectionError as e:
            logger.error(f"Failed to connect Kafka producer: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during Kafka producer connection: {e}")
            raise

    async def disconnect(self) -> None:
        logger.info("Disconnecting Kafka client.")
        if self._producer:
            try:
                logger.debug("Stopping Kafka producer...")
                await self._producer.stop()
                logger.info("Kafka producer stopped.")
            except Exception as e:
                logger.error(f"Error stopping Kafka producer: {e}")
        if self._consumer:
            try:
                logger.debug("Stopping Kafka consumer...")
                await self._consumer.stop()
                logger.info("Kafka consumer stopped.")
            except Exception as e:
                logger.error(f"Error stopping Kafka consumer: {e}")
        if self._reader_task and not self._reader_task.done():
            logger.debug("Cancelling Kafka reader task...")
            self._reader_task.cancel()
            try:
                await self._reader_task
                logger.debug("Kafka reader task cancelled.")
            except asyncio.CancelledError:
                logger.info("Kafka reader task was cancelled as expected.")
            except Exception as e:
                logger.error(f"Error during Kafka reader task cancellation: {e}")

    async def publish(self, topic: str, payload: BaseMessage, **kwargs) -> None:
        if not self._producer:
            logger.error("Kafka producer is not connected. Cannot publish.")
            raise RuntimeError("Kafka producer is not connected.")

        data = payload.model_dump_json().encode()
        logger.debug(
            f"Publishing message to Kafka topic '{topic}'. Payload: {payload.model_dump_json()}"
        )
        try:
            await self._producer.send_and_wait(topic, data)
            logger.info(f"Message successfully sent to Kafka topic '{topic}'.")
        except KafkaTimeoutError as e:
            logger.error(f"Timeout sending message to Kafka topic '{topic}': {e}")
            raise
        except KafkaConnectionError as e:
            logger.error(f"Connection error sending message to Kafka topic '{topic}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error sending message to Kafka topic '{topic}': {e}")
            raise

    async def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        ready_event: Optional[asyncio.Event] = None,
        **kwargs,
    ) -> None:
        """Subscribe to a Kafka topic.

        Args:
            topic: Topic to subscribe to.
            handler: Callback function to handle received messages.
            ready_event: Optional asyncio.Event to signal when consumer is ready.
            **kwargs: Additional keyword arguments for AIOKafkaConsumer
                      (e.g., group_id, auto_offset_reset).
        """
        # if not AIOKAFKA_INSTALLED:
        #     logger.error("aiokafka is not installed. Cannot subscribe to Kafka topic.")
        #     return

        servers = f"{self.config.url}:{self.config.port}"
        listener_instance = None
        if ready_event:
            listener_instance = _KafkaReadyListener(ready_event, topic)

        if self._consumer is None:
            # Default Kafka consumer settings (can be overridden by kwargs)
            consumer_defaults = {
                "group_id": f"default-group-{uuid.uuid4().hex}",  # Unique default group_id
                "auto_offset_reset": "earliest",
                "session_timeout_ms": 30000,  # Default: 30000
                "heartbeat_interval_ms": 10000,  # Default: 10000, must be lower than session_timeout
                "max_poll_interval_ms": 300000,  # Default: 300000
            }
            consumer_config = {**consumer_defaults, **kwargs}
            logger.info(
                f"Creating Kafka consumer for topic '{topic}' with effective config: {consumer_config}"
            )
            try:
                # Explicitly pass the running event loop
                loop = asyncio.get_running_loop()
                self._consumer = AIOKafkaConsumer(
                    bootstrap_servers=servers,
                    loop=loop,  # Pass loop
                    **consumer_config,  # Pass merged config
                )
                await self._consumer.start()
                logger.info(f"Kafka consumer started for topic '{topic}'. Subscribing...")
                # Pass listener if provided
                self._consumer.subscribe(
                    [topic], listener=listener_instance
                )  # This call is synchronous
                logger.info(f"Successfully subscribed to initial topic '{topic}'.")
                if self._reader_task is None or self._reader_task.done():
                    self._reader_task = asyncio.create_task(self._reader())
                    logger.info("Kafka reader task started.")
            except KafkaConnectionError as e:
                logger.error(f"Failed to connect Kafka consumer: {e}")
                # Consider how to handle this error (e.g., raise or return status)
                return  # Or raise specific exception
            except Exception as e:
                logger.error(f"An unexpected error occurred during Kafka consumer setup: {e}")
                return  # Or raise

        else:
            # Consumer already exists, add subscription if not already subscribed
            current_subscription = self._consumer.subscription()
            if topic not in current_subscription:
                new_topic_list = list(current_subscription) + [topic]
                logger.info(
                    f"Adding subscription for topic '{topic}'. Current: {current_subscription}, New: {new_topic_list}"
                )
                try:
                    # Pass listener if provided and applicable for re-subscribe
                    self._consumer.subscribe(  # This call is synchronous
                        topics=new_topic_list, listener=listener_instance
                    )
                    logger.info(f"Successfully added subscription for topic '{topic}'.")
                except Exception as e:
                    logger.error(f"Failed to add subscription for topic '{topic}': {e}")
                    return  # Or raise

        # Register handler
        if topic not in self._listeners:
            self._listeners[topic] = []
        if handler not in self._listeners[topic]:
            self._listeners[topic].append(handler)
            logger.info(f"Handler registered for topic '{topic}'.")

    async def _reader(self) -> None:
        """Single reader for all subscribed topics"""
        if not self._consumer:
            logger.error("Kafka consumer is not initialized. Reader cannot start.")
            return

        logger.info("Kafka message reader task started.")
        try:
            async for msg in self._consumer:
                topic = msg.topic
                partition = msg.partition
                offset = msg.offset
                logger.debug(
                    f"Received message from Kafka. Topic: {topic}, Partition: {partition}, Offset: {offset}, Key: {msg.key}, Timestamp: {msg.timestamp}"
                )

                # Log the raw message value before attempting to decode/process
                try:
                    # Attempt to decode for logging, but handle potential errors
                    decoded_value_for_log = msg.value.decode("utf-8")
                    logger.debug(f"Raw message value (decoded for log): {decoded_value_for_log}")
                except UnicodeDecodeError:
                    logger.debug(
                        f"Raw message value (bytes, could not decode as UTF-8): {msg.value!r}"
                    )
                except Exception as e_dec:
                    logger.warning(f"Could not decode message value for logging: {e_dec}")

                handlers = self._listeners.get(topic, [])
                if not handlers:
                    logger.warning(
                        f"No handlers registered for topic '{topic}'. Discarding message from Partition: {partition}, Offset: {offset}."
                    )
                    continue

                logger.debug(
                    f"Processing message from topic '{topic}' with {len(handlers)} handler(s)."
                )
                for i, h in enumerate(handlers):
                    handler_name = h.__name__ if hasattr(h, "__name__") else str(h)
                    logger.debug(
                        f"Invoking handler {i+1}/{len(handlers)} ('{handler_name}') for topic '{topic}'."
                    )
                    try:
                        await h(msg.value)  # Pass raw bytes as per existing logic
                        logger.debug(
                            f"Handler '{handler_name}' completed for message from topic '{topic}'."
                        )
                    except Exception as e:
                        logger.error(
                            f"Handler '{handler_name}' for topic '{topic}' raised an exception: {e}",
                            exc_info=True,
                        )
        except asyncio.CancelledError:
            logger.info("Kafka reader task was cancelled.")
        except KafkaConnectionError as e:
            logger.error(f"Kafka connection error in reader task: {e}", exc_info=True)
            # Potentially try to reconnect or signal an issue, for now just log and exit task
        except Exception as e:
            logger.error(f"Unexpected error in Kafka reader task: {e}", exc_info=True)
        finally:
            logger.info("Kafka message reader task finished.")

    def is_connected(self) -> bool:
        return self._producer is not None and not self._producer._closed
