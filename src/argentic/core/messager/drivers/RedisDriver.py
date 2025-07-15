from argentic.core.messager.drivers.base_definitions import BaseDriver, MessageHandler
from argentic.core.protocol.message import BaseMessage
from .configs import RedisDriverConfig

import asyncio
from typing import Optional, List, Dict, Type

try:
    import aioredis
    from aioredis.client import PubSub

    AIOREDIS_INSTALLED = True
except ImportError:
    AIOREDIS_INSTALLED = False
    # Define dummy types for type hinting
    PubSub = type("PubSub", (object,), {})
    aioredis = type(
        "aioredis",
        (object,),
        {
            "Redis": type("Redis", (object,), {}),
            "client": type("client", (object,), {"PubSub": PubSub}),
            "from_url": lambda _: None,  # Dummy function
        },
    )


class RedisDriver(BaseDriver[RedisDriverConfig]):
    def __init__(self, config: RedisDriverConfig):
        if not AIOREDIS_INSTALLED:
            raise ImportError(
                "aioredis is not installed. "
                "Please install it with: uv pip install argentic[redis]"
            )
        super().__init__(config)
        self._redis: Optional["aioredis.Redis"] = None
        # topic to list of handlers
        self._listeners: Dict[str, List[MessageHandler]] = {}
        self._pubsub: Optional[PubSub] = None
        self._reader_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        url = f"redis://{self.config.url}:{self.config.port}"
        self._redis = await aioredis.from_url(
            url,
            password=self.config.password,
        )
        return True

    async def disconnect(self) -> None:
        if self._pubsub:
            await self._pubsub.close()
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._redis:
            await self._redis.close()

        self._redis = None
        self._pubsub = None
        self._reader_task = None

    async def publish(
        self, topic: str, payload: BaseMessage, qos: int = 0, retain: bool = False
    ) -> None:
        # Handle BaseMessage serialization
        data = payload.model_dump_json()

        if not self._redis:
            raise ConnectionError("Redis client not connected")
        await self._redis.publish(topic, data)

    async def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        message_cls: Type[BaseMessage] = BaseMessage,
        **kwargs,
    ) -> None:
        if not self._redis:
            raise ConnectionError("Redis client not connected")
        # register handler and subscribe on first handler per topic
        if topic not in self._listeners:
            self._listeners[topic] = []
            # initialize pubsub and reader
            if self._pubsub is None:
                self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe(topic)
            if self._reader_task is None or self._reader_task.done():
                self._reader_task = asyncio.create_task(self._reader())
        self._listeners[topic].append(handler)

    async def _reader(self) -> None:
        # single reader for all topics
        if not self._pubsub:
            return
        async for message in self._pubsub.listen():
            if message.get("type") == "message":
                channel_bytes = message.get("channel")
                if not channel_bytes:
                    continue
                channel = channel_bytes.decode()
                data = message.get("data")
                if not data:
                    continue

                try:
                    msg_obj = BaseMessage.model_validate_json(data)
                    for h in self._listeners.get(channel, []):
                        await h(msg_obj)
                except Exception:
                    # Potentially log this error
                    pass

    def is_connected(self) -> bool:
        return bool(self._redis and not getattr(self._redis, "closed", True))

    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a Redis topic."""
        if self._pubsub and topic in self._listeners:
            await self._pubsub.unsubscribe(topic)
            del self._listeners[topic]

    def format_connection_error_details(self, error: Exception) -> Optional[str]:
        """Format Redis-specific connection error details."""
        if "redis" in str(type(error)).lower():
            return f"Redis error: {error}"
        return None
