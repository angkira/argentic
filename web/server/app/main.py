"""Main FastAPI application."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.dependencies import get_runtime_service
from app.routes import agents_router, supervisors_router, workflows_router, config_router
from app.routes.websocket import create_socket_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print("WebSocket server available at /ws/socket.io")
    yield
    # Shutdown
    print("Shutting down...")
    runtime_service = get_runtime_service()
    await runtime_service.cleanup()


# Create FastAPI app
fastapi_app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Web-based visual builder for Argentic AI agents",
    lifespan=lifespan,
)

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include routers
fastapi_app.include_router(agents_router)
fastapi_app.include_router(supervisors_router)
fastapi_app.include_router(workflows_router)
fastapi_app.include_router(config_router)


@fastapi_app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


@fastapi_app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# Wrap FastAPI app with Socket.IO
app = create_socket_app(fastapi_app)
