from core.messager.drivers import BaseDriver, DriverConfig, MessageHandler
from core.protocol.message import BaseMessage, from_payload

import asyncio
from typing import Optional
import aioredis


class RedisDriver(BaseDriver):
    def __init__(self, config: DriverConfig):
        super().__init__(config)
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        url = f"redis://{self.config.url}:{self.config.port}"
        self._redis = await aioredis.from_url(
            url,
            password=self.config.password,
        )

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.close()

    async def publish(self, topic: str, payload: BaseMessage, **kwargs) -> None:
        # Handle BaseMessage serialization
        data = payload.model_dump_json()

        await self._redis.publish(topic, data)

    async def subscribe(self, topic: str, handler: MessageHandler, **kwargs) -> None:
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(topic)

        async def _reader() -> None:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    # Map raw message to BaseMessage
                    protocol_msg = from_payload(topic, message["data"])
                    await handler(protocol_msg)

        asyncio.create_task(_reader())

    def is_connected(self) -> bool:
        return bool(self._redis and not getattr(self._redis, "closed", True))
