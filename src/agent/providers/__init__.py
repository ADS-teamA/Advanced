"""
Provider abstraction for multi-LLM support.

Supports:
- Anthropic Claude (Haiku, Sonnet, Opus) with prompt caching
- OpenAI GPT-5 (Mini, Nano) with function calling
- Google Gemini (2.5 Flash, Pro) with context caching

Usage:
    from src.agent.providers import create_provider

    provider = create_provider(model="gemini-2.5-flash", google_api_key=api_key)
    response = provider.create_message(messages, tools, system, ...)
"""

from .base import BaseProvider, ProviderResponse
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .factory import create_provider

__all__ = [
    "BaseProvider",
    "ProviderResponse",
    "AnthropicProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "create_provider",
]
