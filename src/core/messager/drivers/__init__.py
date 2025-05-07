from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Optional
import importlib

from pydantic import BaseModel, Field
from core.messager.protocols import MessagerProtocol


# Configuration for any driver
class DriverConfig(BaseModel):
    url: str = Field(..., description="Broker hostname or URL")
    port: int = Field(..., description="Broker port")
    user: Optional[str] = Field(None, description="Username for auth")
    password: Optional[str] = Field(None, description="Password for auth")
    token: Optional[str] = Field(None, description="Token for auth, if applicable")


# Base interface for drivers
class BaseDriver(ABC):
    def __init__(self, config: DriverConfig):
        self.config = config

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to broker"""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanly close connection"""
        ...

    @abstractmethod
    async def publish(self, topic: str, payload: Any, **kwargs) -> None:
        """Publish a message"""
        ...

    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[Any], Coroutine], **kwargs) -> None:
        """Subscribe and assign a handler"""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Return connection status"""
        ...


# Mapping of protocols to module paths and driver class names
_DRIVER_INFO = {
    MessagerProtocol.MQTT: ("core.messager.drivers.MQTTDriver", "MQTTDriver"),
    MessagerProtocol.REDIS: ("core.messager.drivers.RedisDriver", "RedisDriver"),
    MessagerProtocol.KAFKA: ("core.messager.drivers.KafkaDriver", "KafkaDriver"),
    MessagerProtocol.RABBITMQ: ("core.messager.drivers.RabbitMQDriver", "RabbitMQDriver"),
}


def create_driver(protocol: MessagerProtocol, config: DriverConfig) -> BaseDriver:
    """Factory: dynamically import only the requested driver"""
    info = _DRIVER_INFO.get(protocol)

    if not info:
        raise ValueError(f"Unsupported protocol: {protocol}")
    module_path, class_name = info
    module = importlib.import_module(module_path)
    driver_cls = getattr(module, class_name)

    return driver_cls(config)
