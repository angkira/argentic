"""
Gemma Provider - Alias for GemmaJAXProvider.

This module provides a simplified import path for the Gemma provider.
"""

from argentic.core.llm.providers.gemma_jax import GemmaJAXProvider

# Provide GemmaProvider as an alias for backward compatibility and simpler imports
GemmaProvider = GemmaJAXProvider

__all__ = ["GemmaProvider", "GemmaJAXProvider"]
