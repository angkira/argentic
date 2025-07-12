from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Optional, Type

from pydantic import BaseModel, Field
from argentic.core.protocol.message import BaseMessage
from argentic.core.messager.protocols import MessagerProtocol


class DriverConfig(BaseModel):
    url: str = Field(..., description="Broker hostname or URL")
    port: int = Field(..., description="Broker port")
    user: Optional[str] = Field(None, description="Username for auth")
    password: Optional[str] = Field(None, description="Password for auth")
    token: Optional[str] = Field(None, description="Token for auth, if applicable")
    client_id: Optional[str] = Field(None, description="Client ID for the connection")
    keepalive: Optional[int] = Field(60, description="Keep-alive interval for the connection")
    virtualhost: Optional[str] = Field(None, description="Virtual host for RabbitMQ")
    group_id: Optional[str] = Field(None, description="Consumer group ID for Kafka")
    auto_offset_reset: Optional[str] = Field(None, description="Offset reset policy for Kafka")


MessageHandler = Callable[[BaseMessage], Coroutine[Any, Any, None]]


class BaseDriver(ABC):
    """Abstract base class for all messaging drivers."""

    def __init__(self, config: DriverConfig):
        self.config = config

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
