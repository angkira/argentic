from typing import Optional, Literal
from pydantic import BaseModel, Field
from aiomqtt.client import ProtocolVersion


class BaseDriverConfig(BaseModel):
    client_id: Optional[str] = None


class MQTTDriverConfig(BaseDriverConfig):
    url: str
    port: int = 1883
    user: Optional[str] = None
    password: Optional[str] = None
    keepalive: int = 60
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
