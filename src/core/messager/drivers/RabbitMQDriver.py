from core.messager.drivers import BaseDriver, DriverConfig, MessageHandler
from core.protocol.message import BaseMessage

from typing import Optional, List, Dict, Any

try:
    import aio_pika

    AIO_PIKA_INSTALLED = True
except ImportError:
    AIO_PIKA_INSTALLED = False
    # Define dummy types for type hinting
    aio_pika = type(
        "aio_pika",
        (object,),
        {
            "RobustConnection": type("RobustConnection", (object,), {}),
            "Channel": type("Channel", (object,), {}),
            "ExchangeType": type("ExchangeType", (object,), {"FANOUT": "fanout"}),
            "Message": type("Message", (object,), {}),
            "IncomingMessage": type("IncomingMessage", (object,), {}),
            "Queue": type("Queue", (object,), {}),
            "connect_robust": lambda _: None,  # Dummy function
        },
    )

import json
import logging


class RabbitMQDriver(BaseDriver):
    def __init__(self, config: DriverConfig):
        if not AIO_PIKA_INSTALLED:
            raise ImportError(
                "aio-pika is not installed. "
                "Please install it with: uv pip install argentic[rabbitmq]"
            )
        super().__init__(config)
        self._connection: Optional[aio_pika.RobustConnection] = None
        self._channel: Optional[aio_pika.Channel] = None
        # topic to list of handlers
        self._listeners: Dict[str, List[MessageHandler]] = {}
        # track queues per topic
        self._queues: Dict[str, aio_pika.Queue] = {}
        self.logger = logging.getLogger("RabbitMQDriver")

    async def connect(self) -> None:
        url = f"amqp://{self.config.user}:{self.config.password}@{self.config.url}:{self.config.port}/"
        self._connection = await aio_pika.connect_robust(url)
        self._channel = await self._connection.channel()
        self.logger.info(f"Connected to RabbitMQ at {self.config.url}:{self.config.port}")

    async def disconnect(self) -> None:
        if self._connection:
            await self._connection.close()
            self.logger.info("Disconnected from RabbitMQ")

    async def publish(self, topic: str, payload: Any, **kwargs) -> None:
        try:
            exchange = await self._channel.declare_exchange(topic, aio_pika.ExchangeType.FANOUT)

            # Handle BaseMessage serialization with multiple fallback methods
            if isinstance(payload, BaseMessage):
                try:
                    body = payload.model_dump_json().encode("utf-8")
                except Exception as e:
                    self.logger.warning(f"Failed to serialize with model_dump_json: {e}")
                    try:
                        body = json.dumps(payload.model_dump()).encode("utf-8")
                    except Exception as e:
                        self.logger.warning(f"Failed to serialize with model_dump: {e}")
                        # Last resort serialization
                        body = json.dumps(
                            {
                                "id": str(payload.id),
                                "timestamp": (
                                    payload.timestamp.isoformat() if payload.timestamp else None
                                ),
                                "type": payload.__class__.__name__,
                                **{
                                    k: v
                                    for k, v in payload.__dict__.items()
                                    if not k.startswith("_")
                                },
                            }
                        ).encode("utf-8")
            elif isinstance(payload, str):
                body = payload.encode("utf-8")
            elif isinstance(payload, bytes):
                body = payload
            else:
                body = json.dumps(payload).encode("utf-8")

            message = aio_pika.Message(body=body)
            await exchange.publish(message, routing_key="")
            self.logger.debug(f"Published message to exchange: {topic}")
        except Exception as e:
            self.logger.error(f"Error publishing to exchange {topic}: {e}")
            raise

    async def subscribe(self, topic: str, handler: MessageHandler, **kwargs) -> None:
        try:
            # register handler and setup consumer on first subscribe per topic
            if topic not in self._listeners:
                self._listeners[topic] = []
                # declare exchange and queue
                exchange = await self._channel.declare_exchange(topic, aio_pika.ExchangeType.FANOUT)
                queue = await self._channel.declare_queue(exclusive=True)
                await queue.bind(exchange)
                self._queues[topic] = queue
                self.logger.info(f"Created queue for topic: {topic}")

                # single reader for this topic
                async def _reader(message: aio_pika.IncomingMessage) -> None:
                    try:
                        async with message.process():
                            self.logger.debug(f"Received message for topic {topic}")
                            for h in self._listeners.get(topic, []):
                                try:
                                    await h(message.body)
                                except Exception as e:
                                    self.logger.error(f"Handler error for topic {topic}: {e}")
                    except Exception as e:
                        self.logger.error(f"Error processing message for topic {topic}: {e}")

                # Start consumer
                await queue.consume(_reader)
                self.logger.info(f"Started consumer for topic: {topic}")

            self._listeners[topic].append(handler)
            self.logger.info(
                f"Added handler for topic: {topic}, total handlers: {len(self._listeners[topic])}"
            )
        except Exception as e:
            self.logger.error(f"Error subscribing to topic {topic}: {e}")
            raise

    def is_connected(self) -> bool:
        return bool(self._connection and not getattr(self._connection, "is_closed", True))
