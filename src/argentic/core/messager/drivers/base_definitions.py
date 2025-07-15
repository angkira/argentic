from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Optional, Type, TypeVar, Generic

from pydantic import BaseModel
from argentic.core.protocol.message import BaseMessage


MessageHandler = Callable[[BaseMessage], Coroutine[Any, Any, None]]

T_DriverConfig = TypeVar("T_DriverConfig", bound=BaseModel)


class BaseDriver(ABC, Generic[T_DriverConfig]):
    """Abstract base class for all messaging drivers."""

    def __init__(self, config: T_DriverConfig):
        self.config: T_DriverConfig = config

    @abstractmethod
    async def connect(self) -> bool:
        """Initialize connection to broker"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker"""
        pass

    @abstractmethod
    async def publish(
        self, topic: str, payload: BaseMessage, qos: int = 0, retain: bool = False
    ) -> None:
        """Abstract method to publish a message."""
        pass

    @abstractmethod
    async def subscribe(
        self, topic: str, handler: MessageHandler, message_cls: Type[BaseMessage]
    ) -> None:
        """Abstract method to subscribe to a topic."""
        pass

    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        """Abstract method to unsubscribe from a topic."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Checks if the driver is currently connected."""
        pass

    @abstractmethod
    def format_connection_error_details(self, error: Exception) -> Optional[str]:
        """Formats driver-specific connection error details."""
        pass
