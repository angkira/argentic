"""API routes for configuration management."""

from typing import List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.models import LLMProviderConfig, MessagingConfig, LLMProviderType, MessagingProtocol

router = APIRouter(prefix="/api/config", tags=["configuration"])


class LLMProviderInfo(BaseModel):
    """Information about available LLM providers."""

    name: str
    display_name: str
    description: str
    required_fields: List[str]


class MessagingProtocolInfo(BaseModel):
    """Information about available messaging protocols."""

    name: str
    display_name: str
    description: str
    default_port: int


@router.get("/llm-providers", response_model=List[LLMProviderInfo])
async def list_llm_providers():
    """List available LLM providers."""
    return [
        {
            "name": "ollama",
            "display_name": "Ollama",
            "description": "Local model execution via Ollama service",
            "required_fields": ["ollama_model_name", "ollama_base_url"],
        },
        {
            "name": "llama_cpp_server",
            "display_name": "Llama.cpp Server",
            "description": "HTTP-based llama.cpp server integration",
            "required_fields": ["llama_cpp_server_host", "llama_cpp_server_port"],
        },
        {
            "name": "llama_cpp_cli",
            "display_name": "Llama.cpp CLI",
            "description": "Direct CLI binary execution",
            "required_fields": ["llama_cpp_cli_binary", "llama_cpp_cli_model_path"],
        },
        {
            "name": "google_gemini",
            "display_name": "Google Gemini",
            "description": "Cloud-based model via Google API",
            "required_fields": ["google_gemini_api_key", "google_gemini_model_name"],
        },
        {
            "name": "mock",
            "display_name": "Mock (Testing)",
            "description": "Mock provider for testing",
            "required_fields": [],
        },
    ]


@router.get("/messaging-protocols", response_model=List[MessagingProtocolInfo])
async def list_messaging_protocols():
    """List available messaging protocols."""
    return [
        {
            "name": "mqtt",
            "display_name": "MQTT",
            "description": "Lightweight messaging protocol (default)",
            "default_port": 1883,
        },
        {
            "name": "rabbitmq",
            "display_name": "RabbitMQ",
            "description": "Advanced message queueing protocol",
            "default_port": 5672,
        },
        {
            "name": "kafka",
            "display_name": "Apache Kafka",
            "description": "Distributed event streaming platform",
            "default_port": 9092,
        },
        {
            "name": "redis",
            "display_name": "Redis",
            "description": "In-memory data structure store",
            "default_port": 6379,
        },
    ]


@router.post("/llm-providers/validate", response_model=dict)
async def validate_llm_config(config: LLMProviderConfig):
    """Validate LLM provider configuration."""
    # Basic validation - check required fields based on provider
    required_fields_map = {
        LLMProviderType.OLLAMA: ["ollama_model_name"],
        LLMProviderType.LLAMA_CPP_SERVER: ["llama_cpp_server_host", "llama_cpp_server_port"],
        LLMProviderType.LLAMA_CPP_CLI: ["llama_cpp_cli_binary", "llama_cpp_cli_model_path"],
        LLMProviderType.GOOGLE_GEMINI: ["google_gemini_api_key", "google_gemini_model_name"],
    }

    required_fields = required_fields_map.get(config.provider, [])
    missing_fields = []

    for field in required_fields:
        value = getattr(config, field, None)
        if value is None:
            missing_fields.append(field)

    if missing_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required fields: {', '.join(missing_fields)}",
        )

    return {"valid": True, "provider": config.provider}


@router.post("/messaging/validate", response_model=dict)
async def validate_messaging_config(config: MessagingConfig):
    """Validate messaging configuration."""
    # Basic validation
    if not config.broker_address:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Broker address is required",
        )

    if config.port < 1 or config.port > 65535:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Port must be between 1 and 65535",
        )

    return {"valid": True, "protocol": config.protocol}
