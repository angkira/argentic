from core.messager.drivers import BaseDriver, DriverConfig, MessageHandler
from core.protocol.message import BaseMessage

try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
    from aiokafka.errors import KafkaConnectionError, KafkaTimeoutError

    AIOKAFKA_INSTALLED = True
except ImportError:
    AIOKAFKA_INSTALLED = False
    # Define dummy types for type hinting if aiokafka is not installed
    AIOKafkaConsumer = type("AIOKafkaConsumer", (object,), {})
    AIOKafkaProducer = type("AIOKafkaProducer", (object,), {})
    TopicPartition = type("TopicPartition", (object,), {})
    KafkaConnectionError = type("KafkaConnectionError", (Exception,), {})
    KafkaTimeoutError = type("KafkaTimeoutError", (Exception,), {})

import asyncio
import logging


class KafkaDriver(BaseDriver):
    def __init__(self, config: DriverConfig):
        if not AIOKAFKA_INSTALLED:
            raise ImportError(
                "aiokafka is not installed. "
                "Please install it with: uv pip install argentic[kafka]"
            )
        super().__init__(config)
        self._producer: Optional[AIOKafkaProducer] = None
        self._consumer: Optional[AIOKafkaConsumer] = None
        # topic to list of handlers
        self._listeners: Dict[str, List[MessageHandler]] = {}
        # task reading from all subscribed topics
        self._reader_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        servers = f"{self.config.url}:{self.config.port}"
        self._producer = AIOKafkaProducer(bootstrap_servers=servers)
        await self._producer.start()

    async def disconnect(self) -> None:
        if self._producer:
            await self._producer.stop()
        if self._consumer:
            await self._consumer.stop()

    async def publish(self, topic: str, payload: BaseMessage, **kwargs) -> None:
        # Handle BaseMessage serialization
        data = payload.model_dump_json().encode()

        await self._producer.send_and_wait(topic, data)

    async def subscribe(self, topic: str, handler: MessageHandler, **kwargs) -> None:
        # register handler and (re)subscribe consumer
        if topic not in self._listeners:
            self._listeners[topic] = []
            # initialize consumer on first subscribe
            if self._consumer is None:
                servers = f"{self.config.url}:{self.config.port}"
                try:
                    self._consumer = AIOKafkaConsumer(
                        bootstrap_servers=servers,
                        group_id=kwargs.get("group_id", "default-group"),
                        auto_offset_reset=kwargs.get("auto_offset_reset", "earliest"),
                        # Set reasonable defaults for better reliability
                        session_timeout_ms=30000,
                        heartbeat_interval_ms=10000,
                        max_poll_interval_ms=300000,
                    )
                    # subscribe to first topic
                    await self._consumer.start()
                    await self._consumer.subscribe([topic])
                    # start reader task
                    self._reader_task = asyncio.create_task(self._reader())
                except Exception as e:
                    # Log error and reset consumer to None so it can be recreated on next try
                    logger = logging.getLogger("KafkaDriver")
                    logger.error(f"Failed to create Kafka consumer: {e}")
                    self._consumer = None
                    # Re-raise to let the caller know
                    raise
            else:
                try:
                    # add new topic subscription
                    current = list(self._listeners.keys()) + [topic]
                    await self._consumer.subscribe(current)
                except Exception as e:
                    logger = logging.getLogger("KafkaDriver")
                    logger.error(f"Failed to subscribe to topic {topic}: {e}")
                    raise
        self._listeners[topic].append(handler)

    async def _reader(self) -> None:
        """Single reader for all subscribed topics"""
        logger = logging.getLogger("KafkaDriver")

        try:
            logger.info("Starting Kafka message reader")
            async for msg in self._consumer:
                try:
                    # Extract topic and find handlers
                    topic = msg.topic
                    handlers = self._listeners.get(topic, [])
                    if not handlers:
                        logger.debug(f"No handlers for topic {topic}")
                        continue

                    # Process message with all registered handlers
                    logger.debug(f"Processing message from topic {topic}")
                    for h in handlers:
                        try:
                            await h(msg.value)
                        except Exception as e:
                            logger.error(f"Handler error for topic {topic}: {e}")
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")
        except Exception as e:
            logger.error(f"Kafka reader loop terminated: {e}")
            # Don't raise - we want to keep the task alive

    def is_connected(self) -> bool:
        return bool(self._producer and not getattr(self._producer, "_closed", True))
