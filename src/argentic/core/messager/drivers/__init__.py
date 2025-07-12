from typing import Any, Callable, Coroutine
import importlib

from argentic.core.messager.protocols import MessagerProtocol
from argentic.core.protocol.message import BaseMessage

from .base_definitions import (
    BaseDriver,
    DriverConfig,
    MessageHandler,
)  # Import for __all__ and create_driver type hints


# Mapping of protocols to module paths and driver class names
_DRIVER_INFO = {
    MessagerProtocol.MQTT: ("argentic.core.messager.drivers.MQTTDriver", "MQTTDriver"),
    MessagerProtocol.REDIS: ("argentic.core.messager.drivers.RedisDriver", "RedisDriver"),
    MessagerProtocol.KAFKA: ("argentic.core.messager.drivers.KafkaDriver", "KafkaDriver"),
    MessagerProtocol.RABBITMQ: ("argentic.core.messager.drivers.RabbitMQDriver", "RabbitMQDriver"),
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


__all__ = [
    "create_driver",
    "BaseDriver",  # Explicitly export BaseDriver, DriverConfig, MessageHandler
    "DriverConfig",
    "MessageHandler",
]
