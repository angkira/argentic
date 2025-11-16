"""LLM provider configuration models."""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class LLMProviderType(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    LLAMA_CPP_SERVER = "llama_cpp_server"
    LLAMA_CPP_CLI = "llama_cpp_cli"
    GOOGLE_GEMINI = "google_gemini"
    MOCK = "mock"


class LLMProviderConfig(BaseModel):
    """Configuration for LLM providers."""

    provider: LLMProviderType = Field(..., description="LLM provider type")

    # Ollama config
    ollama_model_name: Optional[str] = Field(None, description="Ollama model name")
    ollama_base_url: Optional[str] = Field(
        "http://localhost:11434", description="Ollama base URL"
    )

    # Llama.cpp server config
    llama_cpp_server_host: Optional[str] = Field("localhost", description="Llama.cpp server host")
    llama_cpp_server_port: Optional[int] = Field(8080, description="Llama.cpp server port")
    llama_cpp_server_auto_start: Optional[bool] = Field(
        False, description="Auto-start llama.cpp server"
    )

    # Llama.cpp CLI config
    llama_cpp_cli_binary: Optional[str] = Field(None, description="Path to llama.cpp binary")
    llama_cpp_cli_model_path: Optional[str] = Field(None, description="Path to GGUF model file")

    # Google Gemini config
    google_gemini_api_key: Optional[str] = Field(None, description="Google Gemini API key")
    google_gemini_model_name: Optional[str] = Field(
        "gemini-2.0-flash", description="Gemini model name"
    )

    # Common parameters
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 2048,
        },
        description="LLM parameters (temperature, top_p, etc.)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "google_gemini",
                "google_gemini_api_key": "your-api-key",
                "google_gemini_model_name": "gemini-2.0-flash",
                "parameters": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_output_tokens": 2048,
                },
            }
        }
