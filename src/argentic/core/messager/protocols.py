from enum import Enum


class MessagerProtocol(str, Enum):
    MQTT = "mqtt"
    REDIS = "redis"
    KAFKA = "kafka"
    RABBITMQ = "rabbitmq"
