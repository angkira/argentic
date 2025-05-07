from core.messager.drivers import BaseDriver, DriverConfig
from core.protocol.message import BaseMessage, from_payload


import json
from typing import Any, Callable, Coroutine, Optional
import aio_pika


class RabbitMQDriver(BaseDriver):
    def __init__(self, config: DriverConfig):
        super().__init__(config)
        self._connection: Optional[aio_pika.RobustConnection] = None
        self._channel: Optional[aio_pika.Channel] = None

    async def connect(self) -> None:
        url = f"amqp://{self.config.user}:{self.config.password}@{self.config.url}:{self.config.port}/"
        self._connection = await aio_pika.connect_robust(url)
        self._channel = await self._connection.channel()

    async def disconnect(self) -> None:
        if self._connection:
            await self._connection.close()

    async def publish(self, topic: str, payload: Any, **kwargs) -> None:
        exchange = await self._channel.declare_exchange(topic, aio_pika.ExchangeType.FANOUT)
        # Handle BaseMessage serialization
        if isinstance(payload, BaseMessage):
            body = payload.model_dump_json().encode()
        else:
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        message = aio_pika.Message(body=body)
        await exchange.publish(message, routing_key="")

    async def subscribe(self, topic: str, handler: Callable[[Any], Coroutine], **kwargs) -> None:
        exchange = await self._channel.declare_exchange(topic, aio_pika.ExchangeType.FANOUT)
        queue = await self._channel.declare_queue(exclusive=True)
        await queue.bind(exchange)

        async def _reader(message: aio_pika.IncomingMessage) -> None:
            async with message.process():
                # Map raw RabbitMQ message to BaseMessage
                protocol_msg = from_payload(topic, message.body)
                await handler(protocol_msg)

        await queue.consume(_reader)

    def is_connected(self) -> bool:
        return bool(self._connection and not getattr(self._connection, "is_closed", True))
