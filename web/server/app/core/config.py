"""Application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    app_name: str = "Argentic Web Server"
    app_version: str = "0.1.0"
    debug: bool = False

    # CORS settings
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_prefix = "ARGENTIC_WEB_"
        case_sensitive = False


settings = Settings()
