from typing import Any, Callable, Coroutine, Dict, Type
import importlib

from argentic.core.messager.protocols import MessagerProtocol
from argentic.core.protocol.message import BaseMessage

from .base_definitions import (
    BaseDriver,
    MessageHandler,
)
from .configs import (
    MQTTDriverConfig,
    RedisDriverConfig,
    KafkaDriverConfig,
    RabbitMQDriverConfig,
    BaseDriverConfig,
    DriverConfig,
)


_DRIVER_MAPPING: Dict[MessagerProtocol, tuple[str, Type[BaseDriverConfig]]] = {
    MessagerProtocol.MQTT: ("MQTTDriver", MQTTDriverConfig),
    MessagerProtocol.REDIS: ("RedisDriver", RedisDriverConfig),
    MessagerProtocol.KAFKA: ("KafkaDriver", KafkaDriverConfig),
    MessagerProtocol.RABBITMQ: ("RabbitMQDriver", RabbitMQDriverConfig),
}

_DRIVER_MODULES = {
    "MQTTDriver": "argentic.core.messager.drivers.MQTTDriver",
    "RedisDriver": "argentic.core.messager.drivers.RedisDriver",
    "KafkaDriver": "argentic.core.messager.drivers.KafkaDriver",
    "RabbitMQDriver": "argentic.core.messager.drivers.RabbitMQDriver",
}


def create_driver(protocol: MessagerProtocol, config_data: dict) -> BaseDriver:
    """Factory: dynamically import only the requested driver"""
    mapping = _DRIVER_MAPPING.get(protocol)

    if not mapping:
        raise ValueError(f"Unsupported protocol: {protocol}")

    driver_name, config_class = mapping
    module_path = _DRIVER_MODULES[driver_name]
    try:
        module = importlib.import_module(module_path)
        driver_cls = getattr(module, driver_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not import or find {driver_name} from {module_path}. "
            f"Ensure the driver and its dependencies are correctly installed."
        ) from e

    config = config_class(**config_data)
    return driver_cls(config)


__all__ = [
    "create_driver",
    "BaseDriver",
    "MessageHandler",
    "BaseDriverConfig",
    "MQTTDriverConfig",
    "RedisDriverConfig",
    "KafkaDriverConfig",
    "RabbitMQDriverConfig",
    "DriverConfig",
]
