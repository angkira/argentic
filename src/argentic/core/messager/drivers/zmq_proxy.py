"""ZeroMQ proxy manager for XPUB/XSUB message routing.

This module provides a proxy manager that runs a ZeroMQ XPUB/XSUB device
to enable pub/sub messaging between agents without a centralized broker.
"""

import logging
import threading
import time
from typing import Optional

try:
    import zmq
except ImportError:
    raise ImportError(
        "pyzmq is required for ZeroMQ driver. "
        "Install it with: pip install argentic[zeromq]"
    )

logger = logging.getLogger(__name__)


class ZMQProxyManager:
    """Manages ZeroMQ XPUB/XSUB proxy device for message routing.

    The proxy acts as a central routing point for ZeroMQ pub/sub:
    - Frontend (XSUB): Receives messages from publishers
    - Backend (XPUB): Forwards messages to subscribers

    Supports both embedded (auto-start) and external (user-managed) modes.
    """

    def __init__(self, frontend_url: str, backend_url: str):
        """Initialize proxy manager.

        Args:
            frontend_url: Frontend socket URL (e.g., "tcp://127.0.0.1:5555")
            backend_url: Backend socket URL (e.g., "tcp://127.0.0.1:5556")
        """
        self._frontend_url = frontend_url
        self._backend_url = backend_url
        self._context: Optional[zmq.Context] = None
        self._proxy_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_requested = False

    @property
    def is_running(self) -> bool:
        """Check if proxy is currently running."""
        return self._running

    def start(self) -> None:
        """Start proxy in background thread.

        If already running, this is a no-op.
        Blocks briefly (100ms) to ensure proxy is ready.
        """
        if self._running:
            logger.debug("Proxy already running, skipping start")
            return

        logger.info(
            f"Starting ZeroMQ proxy: frontend={self._frontend_url}, "
            f"backend={self._backend_url}"
        )

        self._stop_requested = False
        self._proxy_thread = threading.Thread(
            target=self._run_proxy, daemon=True, name="zmq-proxy"
        )
        self._proxy_thread.start()

        # Wait briefly for proxy to bind sockets
        time.sleep(0.1)
        logger.info("ZeroMQ proxy started successfully")

    def _run_proxy(self) -> None:
        """Proxy loop - runs in background thread.

        Creates XSUB (frontend) and XPUB (backend) sockets and forwards
        messages between them using zmq.proxy(). Blocks until context
        is terminated.
        """
        try:
            self._context = zmq.Context()
            frontend = self._context.socket(zmq.XSUB)
            backend = self._context.socket(zmq.XPUB)

            frontend.bind(self._frontend_url)
            backend.bind(self._backend_url)

            self._running = True
            logger.debug("Proxy thread entering zmq.proxy() loop")

            # This blocks until context is terminated
            zmq.proxy(frontend, backend)

        except zmq.ContextTerminated:
            logger.debug("Proxy context terminated, stopping gracefully")
        except Exception as e:
            logger.error(f"Proxy error: {e}", exc_info=True)
        finally:
            self._running = False
            if self._context:
                try:
                    frontend.close(linger=0)
                    backend.close(linger=0)
                except Exception as e:
                    logger.debug(f"Error closing proxy sockets: {e}")

    def stop(self) -> None:
        """Terminate proxy gracefully.

        Terminates the ZeroMQ context, which will cause zmq.proxy()
        to exit and the thread to finish. Waits up to 2 seconds for
        thread to complete.
        """
        if not self._running:
            logger.debug("Proxy not running, skipping stop")
            return

        logger.info("Stopping ZeroMQ proxy")
        self._stop_requested = True

        if self._context:
            try:
                self._context.term()
            except Exception as e:
                logger.debug(f"Error terminating proxy context: {e}")

        # Wait for thread to finish
        if self._proxy_thread and self._proxy_thread.is_alive():
            self._proxy_thread.join(timeout=2.0)
            if self._proxy_thread.is_alive():
                logger.warning("Proxy thread did not stop within timeout")

        self._running = False
        logger.info("ZeroMQ proxy stopped")

    def __del__(self):
        """Cleanup on garbage collection."""
        if self._running:
            self.stop()
