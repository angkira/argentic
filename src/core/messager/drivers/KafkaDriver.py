from core.messager.drivers import BaseDriver, DriverConfig, MessageHandler
from core.protocol.message import BaseMessage
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from typing import Optional, Dict, List

import asyncio


class KafkaDriver(BaseDriver):
    def __init__(self, config: DriverConfig):
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
                self._consumer = AIOKafkaConsumer(
                    bootstrap_servers=servers,
                    group_id=kwargs.get("group_id"),
                )
                # subscribe to first topic
                await self._consumer.start()
                await self._consumer.subscribe([topic])
                # start reader task
                self._reader_task = asyncio.create_task(self._reader())
            else:
                # add new topic subscription
                current = list(self._listeners.keys()) + [topic]
                await self._consumer.subscribe(current)
        self._listeners[topic].append(handler)

    async def _reader(self) -> None:
        # single reader for all subscribed topics
        async for msg in self._consumer:
            for h in self._listeners.get(msg.topic, []):
                await h(msg.value)

    def is_connected(self) -> bool:
        return bool(self._producer and not getattr(self._producer, "_closed", True))
