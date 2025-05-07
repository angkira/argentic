from core.messager.drivers import BaseDriver, DriverConfig
from core.protocol.message import BaseMessage, from_payload
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from typing import Optional

import asyncio
import json
from typing import Any, Callable, Coroutine, Dict


class KafkaDriver(BaseDriver):
    def __init__(self, config: DriverConfig):
        super().__init__(config)
        self._producer: Optional[AIOKafkaProducer] = None
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._tasks: Dict[str, asyncio.Task] = {}

    async def connect(self) -> None:
        servers = f"{self.config.url}:{self.config.port}"
        self._producer = AIOKafkaProducer(bootstrap_servers=servers)
        await self._producer.start()

    async def disconnect(self) -> None:
        if self._producer:
            await self._producer.stop()
        if self._consumer:
            await self._consumer.stop()

    async def publish(self, topic: str, payload: Any, **kwargs) -> None:
        # Handle BaseMessage serialization
        if isinstance(payload, BaseMessage):
            data = payload.model_dump_json().encode()
        else:
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        await self._producer.send_and_wait(topic, data)

    async def subscribe(self, topic: str, handler: Callable[[Any], Coroutine], **kwargs) -> None:
        servers = f"{self.config.url}:{self.config.port}"
        self._consumer = AIOKafkaConsumer(
            topic, bootstrap_servers=servers, group_id=kwargs.get("group_id")
        )
        await self._consumer.start()

        async def _reader() -> None:
            async for msg in self._consumer:
                # Map raw Kafka message to BaseMessage
                protocol_msg = from_payload(topic, msg.value)
                await handler(protocol_msg)

        task = asyncio.create_task(_reader())
        self._tasks[topic] = task

    def is_connected(self) -> bool:
        return bool(self._producer and not getattr(self._producer, "_closed", True))
