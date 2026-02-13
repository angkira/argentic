from typing import Literal, Optional

from aiomqtt.client import ProtocolVersion
from pydantic import BaseModel, Field


class BaseDriverConfig(BaseModel):
    client_id: Optional[str] = None


class MQTTDriverConfig(BaseDriverConfig):
    url: str
    port: int = 1883
    user: Optional[str] = None
    password: Optional[str] = None
    # Default MQTT keep-alive (seconds). Shadow-ping interval inside
    # the driver will automatically be set to half of this value.
    keepalive: int = 600
    version: ProtocolVersion = ProtocolVersion.V5

    class Config:
        arbitrary_types_allowed = True


class RedisDriverConfig(BaseDriverConfig):
    url: str
    port: int = 6379
    password: Optional[str] = None


class KafkaDriverConfig(BaseDriverConfig):
    url: str
    port: int = 9092
    group_id: Optional[str] = Field(None, description="Consumer group ID for Kafka")
    auto_offset_reset: Optional[str] = Field(
        "earliest", description="Offset reset policy for Kafka"
    )


class RabbitMQDriverConfig(BaseDriverConfig):
    url: str
    port: int = 5672
    user: Optional[str] = None
    password: Optional[str] = None
    virtualhost: Optional[str] = Field("/", description="Virtual host for RabbitMQ")


class ZeroMQDriverConfig(BaseDriverConfig):
    """ZeroMQ driver configuration with proxy support."""

    url: str = "127.0.0.1"
    port: int = 5555  # Frontend port (XSUB - for publishers)
    backend_port: int = 5556  # Backend port (XPUB - for subscribers)

    # Proxy management
    start_proxy: bool = True  # Auto-start proxy if not running
    proxy_mode: Literal["embedded", "external"] = "embedded"

    # ZeroMQ socket options
    high_water_mark: int = 1000  # Message queue limit
    linger: int = 1000  # Socket close wait time (ms)
    connect_timeout: int = 5000  # Connection timeout (ms)
    topic_encoding: str = "utf-8"


# Generic config for tests (legacy compatibility)


class DriverConfig(BaseDriverConfig):
    url: str
    port: int

    user: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    client_id: Optional[str] = None

    keepalive: int = 600
    version: ProtocolVersion = ProtocolVersion.V5

    group_id: Optional[str] = None
    auto_offset_reset: Optional[str] = "earliest"

    virtualhost: Optional[str] = "/"

    class Config:
        extra = "allow"
