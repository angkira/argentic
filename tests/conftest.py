import os
import sys
import types
import warnings

import pytest

# Add the parent directory to sys.path to ensure imports work properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mark all tests in the tests directory as asyncio tests
pytest.importorskip("pytest_asyncio")


# Filter out specific warnings related to async mocks that we can't easily fix
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    # Suppress coroutine warnings from unittest.mock
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="coroutine '.*' was never awaited", category=RuntimeWarning
        )
        yield


# This allows imports like 'from argentic...' to work correctly

# -----------------------------------------------------------------------------
# Global stubs for heavyweight / external libraries so unit tests are offline.
# This module is imported by pytest before any test collection, ensuring that
# providers import these stubs instead of real libraries.
# -----------------------------------------------------------------------------

# --- google.api_core.exceptions ------------------------------------------------
_google_mod = types.ModuleType("google")
_api_core_mod = types.ModuleType("google.api_core")
_ex_mod = types.ModuleType("google.api_core.exceptions")
for _name in [
    "GoogleAPICallError",
    "ResourceExhausted",
    "InvalidArgument",
    "PermissionDenied",
    "InternalServerError",
    "DeadlineExceeded",
    "ServiceUnavailable",
    "BadRequest",
    "NotFound",
    "Unauthenticated",
    "Unknown",
]:
    setattr(_ex_mod, _name, type(_name, (Exception,), {}))
_api_core_mod.exceptions = _ex_mod  # type: ignore[attr-defined]
_google_mod.api_core = _api_core_mod  # type: ignore[attr-defined]

sys.modules.setdefault("google", _google_mod)
sys.modules.setdefault("google.api_core", _api_core_mod)
sys.modules.setdefault("google.api_core.exceptions", _ex_mod)

# -----------------------------------------------------------------------------
# Optional: stub llama_cpp to prevent ImportError's downstream
# -----------------------------------------------------------------------------
_llama_cpp_mod = types.ModuleType("llama_cpp")
sys.modules.setdefault("llama_cpp", _llama_cpp_mod)

# -----------------------------------------------------------------------------
# Stub aiortc for WebRTC driver tests
# -----------------------------------------------------------------------------
_aiortc_mod = types.ModuleType("aiortc")
_aiortc_contrib_mod = types.ModuleType("aiortc.contrib")
_aiortc_contrib_media_mod = types.ModuleType("aiortc.contrib.media")


# Create mock classes
class _MockRTCPeerConnection:
    def __init__(self, configuration=None):
        self.connectionState = "new"
        self.localDescription = None
        self.remoteDescription = None
        self._track_handlers = []

    def on(self, event):
        def decorator(func):
            self._track_handlers.append((event, func))
            return func

        return decorator

    async def setRemoteDescription(self, desc):
        self.remoteDescription = desc

    async def createAnswer(self):
        class _MockDescription:
            sdp = "mock_answer_sdp"
            type = "answer"

        return _MockDescription()

    async def setLocalDescription(self, desc):
        self.localDescription = desc

    async def close(self):
        self.connectionState = "closed"


class _MockRTCSessionDescription:
    def __init__(self, sdp, type):
        self.sdp = sdp
        self.type = type


class _MockMediaRelay:
    pass


_aiortc_mod.RTCPeerConnection = _MockRTCPeerConnection  # type: ignore[attr-defined]
_aiortc_mod.RTCSessionDescription = _MockRTCSessionDescription  # type: ignore[attr-defined]
_aiortc_contrib_media_mod.MediaRelay = _MockMediaRelay  # type: ignore[attr-defined]
_aiortc_contrib_mod.media = _aiortc_contrib_media_mod  # type: ignore[attr-defined]
_aiortc_mod.contrib = _aiortc_contrib_mod  # type: ignore[attr-defined]

sys.modules.setdefault("aiortc", _aiortc_mod)
sys.modules.setdefault("aiortc.contrib", _aiortc_contrib_mod)
sys.modules.setdefault("aiortc.contrib.media", _aiortc_contrib_media_mod)

# -----------------------------------------------------------------------------
# Stub av (PyAV) for video/audio processing
# -----------------------------------------------------------------------------
_av_mod = types.ModuleType("av")


class _MockVideoFrame:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

    def to_ndarray(self, format="rgb24"):
        import numpy as np

        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)


class _MockAudioFrame:
    def __init__(self, samples=1024, sample_rate=16000):
        self.samples = samples
        self.sample_rate = sample_rate

    def to_ndarray(self):
        import numpy as np

        return np.random.randn(self.samples).astype(np.float32)


_av_mod.VideoFrame = _MockVideoFrame  # type: ignore[attr-defined]
_av_mod.AudioFrame = _MockAudioFrame  # type: ignore[attr-defined]

sys.modules.setdefault("av", _av_mod)

# -----------------------------------------------------------------------------
# Stub gemma library for Gemma 3n tests
# -----------------------------------------------------------------------------
_gemma_mod = types.ModuleType("gemma")
_gemma_gm_mod = types.ModuleType("gm")  # This is what "from gemma import gm" imports
_gemma_gm_text_mod = types.ModuleType("gm.text")
_gemma_gm_ckpts_mod = types.ModuleType("gm.ckpts")


class _MockChatSampler:
    def __init__(self, model_name, params, multi_turn=True):
        self.model_name = model_name
        self.params = params
        self.multi_turn = multi_turn

    def sample(self, messages, temperature=0.7, top_p=0.95, max_tokens=2048):
        return f"Mock response for {len(messages)} messages"


def _mock_load_params(checkpoint_path):
    return {"mock_params": True}


_gemma_gm_text_mod.ChatSampler = _MockChatSampler  # type: ignore[attr-defined]
_gemma_gm_ckpts_mod.load_params = _mock_load_params  # type: ignore[attr-defined]
_gemma_gm_mod.text = _gemma_gm_text_mod  # type: ignore[attr-defined]
_gemma_gm_mod.ckpts = _gemma_gm_ckpts_mod  # type: ignore[attr-defined]
_gemma_mod.gm = _gemma_gm_mod  # type: ignore[attr-defined]

sys.modules.setdefault("gemma", _gemma_mod)
sys.modules.setdefault("gemma.gm", _gemma_gm_mod)

# -----------------------------------------------------------------------------
# Stub jax for JAX operations
# -----------------------------------------------------------------------------
_jax_mod = types.ModuleType("jax")
_jax_mod.devices = lambda: [{"platform": "cpu"}]  # type: ignore[attr-defined]

sys.modules.setdefault("jax", _jax_mod)

# No pytest hooks/fixtures yet—stubs done at import time.
