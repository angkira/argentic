from core.messager.drivers import BaseDriver, DriverConfig
from core.protocol.message import BaseMessage


from paho.mqtt.client import topic_matches_sub
import aiomqtt

import asyncio
import contextlib
import json
from typing import Any, Callable, Coroutine, Dict, Optional, List


class MQTTDriver(BaseDriver):
    def __init__(self, config: DriverConfig):
        super().__init__(config)
        self._client = aiomqtt.Client(
            hostname=config.url,
            port=config.port,
            username=config.user,
            password=config.password,
            tls_params=None,
        )
        self._listeners: Dict[str, List[Callable[[BaseMessage], Coroutine]]] = {}
        self._listen_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        await self._client.__aenter__()
        # start background listener
        self._listen_task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        async for msg in self._client.messages:
            for pattern, handlers in self._listeners.items():
                if topic_matches_sub(pattern, msg.topic.value):
                    for handler in handlers:
                        asyncio.create_task(handler(msg.payload))

    async def disconnect(self) -> None:
        if self._listen_task:
            self._listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listen_task
        await self._client.__aexit__(None, None, None)

    async def publish(self, topic: str, payload: Any, qos: int = 0, retain: bool = False) -> None:
        # Handle BaseMessage serialization
        if isinstance(payload, BaseMessage):
            data = payload.model_dump_json().encode()
        else:
            data = payload if isinstance(payload, (bytes, str)) else json.dumps(payload).encode()
        await self._client.publish(topic, payload=data, qos=qos, retain=retain)

    async def subscribe(
        self, topic: str, handler: Callable[[BaseMessage], Coroutine], qos: int = 0
    ) -> None:
        # register handler and subscribe once per topic
        if topic not in self._listeners:
            self._listeners[topic] = []
            await self._client.subscribe(topic, qos=qos)
        self._listeners[topic].append(handler)

    def is_connected(self) -> bool:
        return bool(self._client and getattr(self._client, "_connected", False))
