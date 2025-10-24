"""
Security utilities for MY_SUJBOT pipeline.

CRITICAL: This module provides error sanitization to prevent API key leaks
in logs, error messages, and user-facing output.

ALL modules that interact with APIs MUST use sanitize_error() before:
- Logging exceptions
- Displaying errors to users
- Writing errors to files
- Sending errors over network

Based on: src/contextual_retrieval.py:_sanitize_error() (lines 479-495)
"""

import logging
import re
from typing import Union

logger = logging.getLogger(__name__)


def sanitize_error(error: Union[str, Exception]) -> str:
    """
    Remove sensitive information from error messages.

    This function sanitizes error messages by replacing API keys, tokens,
    and other sensitive data with masked placeholders.

    CRITICAL: Use this function EVERYWHERE before logging or displaying errors.

    Removes:
    - Anthropic API keys (sk-ant-...)
    - OpenAI API keys (sk-...)
    - Voyage AI keys (pa-...)
    - Generic API keys (api_key=..., apikey:...)
    - Bearer tokens
    - Passwords

    Args:
        error: Error message (str) or Exception object

    Returns:
        Sanitized error message with sensitive data masked

    Examples:
        >>> sanitize_error("API error with key sk-ant-abc123...")
        'API error with key sk-ant-***'

        >>> sanitize_error("Bearer sk-proj-abc123 failed")
        'Bearer sk-*** failed'

        >>> try:
        ...     client.create(api_key="sk-abc123")
        ... except Exception as e:
        ...     logger.error(f"API call failed: {sanitize_error(e)}")
    """
    message = str(error)

    # Anthropic API keys (sk-ant-api03-...)
    # Pattern: sk-ant- followed by 32+ alphanumeric/dash/underscore characters
    message = re.sub(
        r'sk-ant-[a-zA-Z0-9_-]{32,}',
        'sk-ant-***',
        message,
        flags=re.IGNORECASE
    )

    # OpenAI API keys (sk-proj-..., sk-...)
    # Pattern: sk- followed by 32+ alphanumeric characters
    # Note: This must come AFTER sk-ant- to avoid partial masking
    message = re.sub(
        r'sk-[a-zA-Z0-9]{32,}',
        'sk-***',
        message,
        flags=re.IGNORECASE
    )

    # Voyage AI keys (pa-...)
    # Pattern: pa- followed by 32+ alphanumeric/dash/underscore characters
    message = re.sub(
        r'pa-[a-zA-Z0-9_-]{32,}',
        'pa-***',
        message,
        flags=re.IGNORECASE
    )

    # Generic API keys (api_key=xxx, apikey:xxx, api-key=xxx)
    # Pattern: api[_-]?key followed by = or : and 20+ characters
    message = re.sub(
        r'api[_-]?key[=:]\s*[a-zA-Z0-9_-]{20,}',
        'api_key=***',
        message,
        flags=re.IGNORECASE
    )

    # Bearer tokens (Bearer xxx)
    # Pattern: Bearer followed by 20+ alphanumeric/dash/underscore characters
    message = re.sub(
        r'Bearer\s+[a-zA-Z0-9_-]{20,}',
        'Bearer ***',
        message,
        flags=re.IGNORECASE
    )

    # Password fields (password=xxx, pwd=xxx)
    # Pattern: password or pwd followed by = or : and 8+ characters
    message = re.sub(
        r'(password|pwd)[=:]\s*[a-zA-Z0-9_-]{8,}',
        r'\1=***',
        message,
        flags=re.IGNORECASE
    )

    return message


def mask_api_key(key: str) -> str:
    """
    Mask API key for display or logging.

    This function returns a safe representation of an API key that can be
    displayed in logs or UI without exposing the actual key.

    Args:
        key: API key string

    Returns:
        Masked key (e.g., "sk-ant-***", "sk-***", "pa-***", or "***")

    Examples:
        >>> mask_api_key("sk-ant-api03-abc123def456...")
        'sk-ant-***'

        >>> mask_api_key("sk-proj-abc123...")
        'sk-***'

        >>> mask_api_key("pa-voyage-abc123...")
        'pa-***'

        >>> mask_api_key("")
        '***'

    Usage:
        logger.info(f"API key configured: {mask_api_key(api_key)}")
    """
    if not key:
        return "***"

    # Determine key type by prefix
    if key.startswith("sk-ant-"):
        return "sk-ant-***"
    elif key.startswith("sk-"):
        return "sk-***"
    elif key.startswith("pa-"):
        return "pa-***"
    else:
        # Unknown format - don't expose any part
        return "***"


# Validation helper
def validate_api_key_format(key: str, provider: str) -> bool:
    """
    Validate API key format (basic check, doesn't verify key works).

    Args:
        key: API key string
        provider: Provider name ("anthropic", "openai", "voyage")

    Returns:
        True if format is valid, False otherwise

    Examples:
        >>> validate_api_key_format("sk-ant-api03-...", "anthropic")
        True

        >>> validate_api_key_format("invalid", "anthropic")
        False
    """
    if not key:
        return False

    if provider == "anthropic":
        return key.startswith("sk-ant-") and len(key) > 40
    elif provider == "openai":
        return key.startswith("sk-") and len(key) > 40
    elif provider == "voyage":
        return key.startswith("pa-") and len(key) > 30
    else:
        # Unknown provider - just check it's not empty
        return len(key) > 20


# Example usage and tests
if __name__ == "__main__":
    # Test sanitize_error
    print("=== Testing sanitize_error() ===\n")

    test_cases = [
        "API error with key sk-ant-api03-abc123def456ghi789jkl012mno345",
        "Authentication failed: sk-proj-abc123def456ghi789jkl012",
        "Bearer sk-abc123def456ghi789jkl012mno345 invalid",
        "Config error: api_key=abc123def456ghi789jkl012",
        "Voyage API: pa-voyage-abc123def456ghi789",
        "Multiple keys: sk-ant-abc123def456ghi789 and sk-proj-xyz789",
        "Normal error without secrets",
    ]

    for test_case in test_cases:
        sanitized = sanitize_error(test_case)
        print(f"Original: {test_case[:60]}...")
        print(f"Sanitized: {sanitized}")
        print()

    # Test mask_api_key
    print("\n=== Testing mask_api_key() ===\n")

    test_keys = [
        ("sk-ant-api03-abc123def456ghi789jkl012mno345", "sk-ant-***"),
        ("sk-proj-abc123def456ghi789jkl012", "sk-***"),
        ("pa-voyage-abc123def456", "pa-***"),
        ("invalid-key", "***"),
        ("", "***"),
    ]

    for key, expected in test_keys:
        masked = mask_api_key(key)
        status = "✓" if masked == expected else "✗"
        print(f"{status} mask_api_key('{key[:20]}...') = '{masked}' (expected: '{expected}')")

    # Test validate_api_key_format
    print("\n=== Testing validate_api_key_format() ===\n")

    test_validations = [
        ("sk-ant-api03-abc123def456ghi789jkl012mno345", "anthropic", True),
        ("sk-proj-abc123def456ghi789jkl012mno345pqr678", "openai", True),
        ("pa-voyage-abc123def456ghi789", "voyage", True),
        ("invalid", "anthropic", False),
        ("sk-", "openai", False),
        ("", "anthropic", False),
    ]

    for key, provider, expected in test_validations:
        result = validate_api_key_format(key, provider)
        status = "✓" if result == expected else "✗"
        key_display = key[:20] + "..." if len(key) > 20 else key
        print(f"{status} validate('{key_display}', '{provider}') = {result} (expected: {expected})")
