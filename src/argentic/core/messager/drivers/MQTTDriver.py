from argentic.core.messager.drivers import BaseDriver, DriverConfig
from argentic.core.protocol.message import BaseMessage


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
        """Publish a message to the MQTT broker with a longer timeout"""
        if isinstance(payload, BaseMessage):
            # Use model_dump first and then json.dumps to ensure proper serialization
            try:
                data = payload.model_dump_json().encode("utf-8")
            except Exception:
                # Fallback to manual JSON serialization via model_dump
                try:
                    data = json.dumps(payload.model_dump()).encode("utf-8")
                except Exception:
                    # Last resort, try direct JSON serialization
                    data = json.dumps(
                        {
                            "id": str(payload.id),
                            "timestamp": (
                                payload.timestamp.isoformat() if payload.timestamp else None
                            ),
                            "type": payload.__class__.__name__,
                            **{k: v for k, v in payload.__dict__.items() if not k.startswith("_")},
                        }
                    ).encode("utf-8")
        elif isinstance(payload, str):
            data = payload.encode("utf-8")
        elif isinstance(payload, bytes):
            data = payload
        else:
            data = json.dumps(payload).encode("utf-8")

        # Increase timeout to 30 seconds and set QoS to 1 for better reliability
        await self._client.publish(topic, payload=data, qos=qos, retain=retain, timeout=30.0)

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
