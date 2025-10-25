"""
Provider abstraction for multi-LLM support.

Supports:
- Anthropic Claude (Haiku, Sonnet, Opus) with prompt caching
- OpenAI GPT-5 (Mini, Nano) with function calling

Usage:
    from src.agent.providers import create_provider

    provider = create_provider(model="gpt-5-mini", api_key=api_key)
    response = provider.create_message(messages, tools, system, ...)
"""

from .base import BaseProvider, ProviderResponse
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .factory import create_provider

__all__ = [
    "BaseProvider",
    "ProviderResponse",
    "AnthropicProvider",
    "OpenAIProvider",
    "create_provider",
]
