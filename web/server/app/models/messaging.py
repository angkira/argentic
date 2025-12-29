"""Messaging configuration models."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class MessagingProtocol(str, Enum):
    """Supported messaging protocols."""

    MQTT = "mqtt"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    REDIS = "redis"


class MessagingConfig(BaseModel):
    """Configuration for messaging layer."""

    protocol: MessagingProtocol = Field(MessagingProtocol.MQTT, description="Messaging protocol")
    broker_address: str = Field("localhost", description="Broker address")
    port: int = Field(1883, description="Broker port")
    client_id: Optional[str] = Field(None, description="Client ID")
    username: Optional[str] = Field(None, description="Username for authentication")
    password: Optional[str] = Field(None, description="Password for authentication")
    keepalive: int = Field(60, description="Keepalive interval in seconds")
    use_tls: bool = Field(False, description="Use TLS/SSL")

    class Config:
        json_schema_extra = {
            "example": {
                "protocol": "mqtt",
                "broker_address": "localhost",
                "port": 1883,
                "keepalive": 60,
                "use_tls": False,
            }
        }
