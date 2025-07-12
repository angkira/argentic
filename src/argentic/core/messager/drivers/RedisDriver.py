from argentic.core.messager.drivers import BaseDriver, DriverConfig, MessageHandler
from argentic.core.protocol.message import BaseMessage

import asyncio
from typing import Optional, List, Dict

try:
    import aioredis

    AIOREDIS_INSTALLED = True
except ImportError:
    AIOREDIS_INSTALLED = False
    # Define dummy types for type hinting
    aioredis = type(
        "aioredis",
        (object,),
        {
            "Redis": type("Redis", (object,), {}),
            "client": type("client", (object,), {"PubSub": type("PubSub", (object,), {})}),
            "from_url": lambda _: None,  # Dummy function
        },
    )


class RedisDriver(BaseDriver):
    def __init__(self, config: DriverConfig):
        if not AIOREDIS_INSTALLED:
            raise ImportError(
                "aioredis is not installed. "
                "Please install it with: uv pip install argentic[redis]"
            )
        super().__init__(config)
        self._redis: Optional[aioredis.Redis] = None
        # topic to list of handlers
        self._listeners: Dict[str, List[MessageHandler]] = {}
        self._pubsub: Optional[aioredis.client.PubSub] = None
        self._reader_task: Optional[asyncio.Task] = None

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
        # register handler and subscribe on first handler per topic
        if topic not in self._listeners:
            self._listeners[topic] = []
            # initialize pubsub and reader
            if self._pubsub is None:
                self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe(topic)
            if self._reader_task is None:
                self._reader_task = asyncio.create_task(self._reader())
        self._listeners[topic].append(handler)

    async def _reader(self) -> None:
        # single reader for all topics
        async for message in self._pubsub.listen():
            if message.get("type") == "message":
                channel = message.get("channel")
                data = message.get("data")
                for h in self._listeners.get(channel, []):
                    await h(data)

    def is_connected(self) -> bool:
        return bool(self._redis and not getattr(self._redis, "closed", True))
