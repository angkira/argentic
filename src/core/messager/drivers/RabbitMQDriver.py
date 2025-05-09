from core.messager.drivers import BaseDriver, DriverConfig, MessageHandler
from core.protocol.message import BaseMessage

from typing import Optional, List, Dict
import aio_pika


class RabbitMQDriver(BaseDriver):
    def __init__(self, config: DriverConfig):
        super().__init__(config)
        self._connection: Optional[aio_pika.RobustConnection] = None
        self._channel: Optional[aio_pika.Channel] = None
        # topic to list of handlers
        self._listeners: Dict[str, List[MessageHandler]] = {}
        # track queues per topic
        self._queues: Dict[str, aio_pika.Queue] = {}

    async def connect(self) -> None:
        url = f"amqp://{self.config.user}:{self.config.password}@{self.config.url}:{self.config.port}/"
        self._connection = await aio_pika.connect_robust(url)
        self._channel = await self._connection.channel()

    async def disconnect(self) -> None:
        if self._connection:
            await self._connection.close()

    async def publish(self, topic: str, payload: BaseMessage, **kwargs) -> None:
        exchange = await self._channel.declare_exchange(topic, aio_pika.ExchangeType.FANOUT)
        # Handle BaseMessage serialization
        body = payload.model_dump_json().encode()

        message = aio_pika.Message(body=body)
        await exchange.publish(message, routing_key="")

    async def subscribe(self, topic: str, handler: MessageHandler, **kwargs) -> None:
        # register handler and setup consumer on first subscribe per topic
        if topic not in self._listeners:
            self._listeners[topic] = []
            # declare exchange and queue
            exchange = await self._channel.declare_exchange(topic, aio_pika.ExchangeType.FANOUT)
            queue = await self._channel.declare_queue(exclusive=True)
            await queue.bind(exchange)
            self._queues[topic] = queue

            # single reader for this topic
            async def _reader(message: aio_pika.IncomingMessage) -> None:
                async with message.process():
                    for h in self._listeners.get(topic, []):
                        await h(message.body)

            await queue.consume(_reader)
        self._listeners[topic].append(handler)

    def is_connected(self) -> bool:
        return bool(self._connection and not getattr(self._connection, "is_closed", True))
