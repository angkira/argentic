#!/usr/bin/env python3
"""
Visual Agent Validation Script

This script validates that all Visual Agent components are properly installed
and can be imported/instantiated without errors.

Run with: python tests/validate_visual_agent.py
"""

import asyncio
import sys
import types
from typing import List, Tuple

# Mock gemma library BEFORE any imports (it's optional for validation)
_gemma_mod = types.ModuleType("gemma")
_gemma_text_mod = types.ModuleType("gemma.text")
_gemma_ckpts_mod = types.ModuleType("gemma.ckpts")


class _MockChatSampler:
    def __init__(self, model_name, params, multi_turn=True):
        pass


_gemma_text_mod.ChatSampler = _MockChatSampler
_gemma_ckpts_mod.load_params = lambda x: {}
_gemma_mod.text = _gemma_text_mod
_gemma_mod.ckpts = _gemma_ckpts_mod

sys.modules.setdefault("gemma", _gemma_mod)
sys.modules.setdefault("gemma.text", _gemma_text_mod)
sys.modules.setdefault("gemma.ckpts", _gemma_ckpts_mod)


def check_imports() -> List[Tuple[str, bool, str]]:
    """Check all required imports."""
    results = []

    # Core imports
    imports_to_test = [
        ("WebRTCDriver", "argentic.core.drivers", "WebRTCDriver"),
        ("VisualAgent", "argentic.core.agent.visual_agent", "VisualAgent"),
        ("GemmaProvider", "argentic.core.llm.providers.gemma", "GemmaProvider"),
        ("Messager", "argentic.core.messager", "Messager"),
        ("LLMFactory", "argentic.core.llm.llm_factory", "LLMFactory"),
        ("ChatMessage", "argentic.core.protocol.chat_message", "UserMessage"),
        ("MultimodalContent", "argentic.core.protocol.chat_message", "MultimodalContent"),
    ]

    for name, module, attr in imports_to_test:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            results.append((name, True, "✅ Import successful"))
        except ImportError as e:
            results.append((name, False, f"❌ Import failed: {e}"))
        except AttributeError as e:
            results.append((name, False, f"❌ Attribute not found: {e}"))

    return results


async def test_instantiation() -> List[Tuple[str, bool, str]]:
    """Test that components can be instantiated."""
    from unittest.mock import AsyncMock

    # Import components (gemma already mocked at module level)
    from argentic.core.drivers import WebRTCDriver
    from argentic.core.agent.visual_agent import VisualAgent
    from argentic.core.llm.providers.gemma import GemmaProvider
    from argentic.core.messager import Messager

    results = []

    # Test WebRTCDriver
    try:
        driver = WebRTCDriver(video_buffer_size=10, frame_rate=5)
        assert driver.video_buffer_size == 10
        assert driver.frame_rate == 5
        results.append(("WebRTCDriver", True, "✅ Instantiation successful"))
    except Exception as e:
        results.append(("WebRTCDriver", False, f"❌ Failed: {e}"))

    # Test GemmaProvider
    try:
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)
        assert provider.model_name == "gemma-3n-e4b-it"
        results.append(("GemmaProvider", True, "✅ Instantiation successful"))
    except Exception as e:
        results.append(("GemmaProvider", False, f"❌ Failed: {e}"))

    # Test VisualAgent
    try:
        messager = AsyncMock(spec=Messager)
        driver = WebRTCDriver()
        provider = GemmaProvider({"gemma_model_name": "test"})

        agent = VisualAgent(
            llm=provider,
            messager=messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )
        assert agent.driver is driver
        results.append(("VisualAgent", True, "✅ Instantiation successful"))
    except Exception as e:
        results.append(("VisualAgent", False, f"❌ Failed: {e}"))

    return results


async def run_validation():
    """Run all validation checks."""
    print("=" * 70)
    print("Visual Agent Validation Report")
    print("=" * 70)
    print()

    # Check imports
    print("1. Import Validation")
    print("-" * 70)
    import_results = check_imports()
    all_imports_ok = all(success for _, success, _ in import_results)

    for name, success, message in import_results:
        status = "✅" if success else "❌"
        print(f"  {status} {name:25} {message}")

    print()

    # Check instantiation
    print("2. Instantiation Validation")
    print("-" * 70)
    inst_results = await test_instantiation()
    all_inst_ok = all(success for _, success, _ in inst_results)

    for name, success, message in inst_results:
        status = "✅" if success else "❌"
        print(f"  {status} {name:25} {message}")

    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    total_checks = len(import_results) + len(inst_results)
    passed_checks = sum(success for _, success, _ in import_results + inst_results)

    print(f"  Total checks:  {total_checks}")
    print(f"  Passed:        {passed_checks}")
    print(f"  Failed:        {total_checks - passed_checks}")
    print()

    if all_imports_ok and all_inst_ok:
        print("  ✨ All validation checks passed!")
        print()
        print("  The Visual Agent implementation is ready to use.")
        print("  For production use, ensure:")
        print("    - WebRTC signaling server is configured")
        print("    - Gemma 3n model checkpoint is downloaded")
        print("    - MQTT broker is running")
        print()
        return 0
    else:
        print("  ⚠️  Some validation checks failed.")
        print("  Please review the errors above.")
        print()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_validation())
    sys.exit(exit_code)
