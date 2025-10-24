"""
Retry decorator with exponential backoff for MY_SUJBOT pipeline.

Provides reusable retry logic for handling transient failures:
- Rate limits (429 errors)
- Network timeouts
- Temporary service unavailability

Replaces duplicated retry loops in:
- contextual_retrieval.py (_generate_with_anthropic, _generate_with_openai)

Usage:
    from src.utils import retry_with_exponential_backoff

    @retry_with_exponential_backoff(
        max_retries=3,
        exceptions=(RateLimitError, APITimeoutError)
    )
    def call_api():
        return client.messages.create(...)
"""

import logging
import time
from functools import wraps
from typing import Callable, Optional, Tuple, Type

from .security import sanitize_error

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
    retry_condition: Optional[Callable[[Exception], bool]] = None
):
    """
    Decorator for exponential backoff retry logic.

    This decorator automatically retries a function on failure with exponentially
    increasing delays between attempts.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 2.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        backoff_factor: Multiplier for delay on each retry (default: 2.0)
        exceptions: Tuple of exception types to retry (default: all exceptions)
        on_retry: Optional callback(exception, attempt, delay) called before each retry
        retry_condition: Optional function(exception) -> bool to decide if retry

    Returns:
        Decorated function that retries on failure

    Raises:
        Original exception after max_retries exhausted

    Examples:
        >>> # Basic usage - retry any exception
        >>> @retry_with_exponential_backoff()
        >>> def flaky_api_call():
        >>>     return client.create(...)

        >>> # Retry only rate limits
        >>> @retry_with_exponential_backoff(
        >>>     exceptions=(RateLimitError,),
        >>>     max_retries=5
        >>> )
        >>> def rate_limited_call():
        >>>     return client.create(...)

        >>> # Custom retry condition (check error message)
        >>> def is_retryable(e):
        >>>     return "rate" in str(e).lower() or "429" in str(e)
        >>>
        >>> @retry_with_exponential_backoff(
        >>>     retry_condition=is_retryable
        >>> )
        >>> def smart_retry():
        >>>     return client.create(...)

        >>> # With retry callback
        >>> def log_retry(exception, attempt, delay):
        >>>     logger.info(f"Retrying after {delay}s (attempt {attempt}): {exception}")
        >>>
        >>> @retry_with_exponential_backoff(on_retry=log_retry)
        >>> def logged_call():
        >>>     return client.create(...)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    # Try executing function
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    # Check if this is the last attempt
                    if attempt == max_retries:
                        logger.warning(
                            f"Function '{func.__name__}' failed after {max_retries} retries. "
                            f"Giving up."
                        )
                        raise

                    # Check retry condition if provided
                    if retry_condition and not retry_condition(e):
                        logger.debug(
                            f"Function '{func.__name__}' failed with non-retryable error: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)

                    # Log retry
                    logger.warning(
                        f"Function '{func.__name__}' failed (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay:.1f}s... Error: {type(e).__name__}: {sanitize_error(str(e))[:100]}"
                    )

                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, attempt + 1, delay)
                        except Exception as callback_error:
                            logger.error(
                                f"Retry callback failed: {sanitize_error(callback_error)}",
                                exc_info=True
                            )

                    # Sleep before retry
                    time.sleep(delay)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def is_rate_limit_error(exception: Exception) -> bool:
    """
    Check if exception is a rate limit error.

    Args:
        exception: Exception to check

    Returns:
        True if exception indicates rate limiting

    Examples:
        >>> is_rate_limit_error(Exception("429 Rate Limit"))
        True

        >>> is_rate_limit_error(Exception("Connection timeout"))
        False
    """
    error_str = str(exception).lower()
    error_type = type(exception).__name__.lower()

    # Check error message
    rate_limit_indicators = ["rate", "429", "quota", "limit"]
    if any(indicator in error_str for indicator in rate_limit_indicators):
        return True

    # Check exception type name
    if "ratelimit" in error_type or "quota" in error_type:
        return True

    return False


def is_timeout_error(exception: Exception) -> bool:
    """
    Check if exception is a timeout error.

    Args:
        exception: Exception to check

    Returns:
        True if exception indicates timeout

    Examples:
        >>> is_timeout_error(Exception("Request timeout"))
        True

        >>> is_timeout_error(Exception("Invalid API key"))
        False
    """
    error_str = str(exception).lower()
    error_type = type(exception).__name__.lower()

    # Check error message
    timeout_indicators = ["timeout", "timed out", "deadline"]
    if any(indicator in error_str for indicator in timeout_indicators):
        return True

    # Check exception type name
    if "timeout" in error_type:
        return True

    return False


def is_retryable_error(exception: Exception) -> bool:
    """
    Check if exception is retryable (rate limit or timeout).

    This is a combined check for common transient errors that benefit from retry.

    Args:
        exception: Exception to check

    Returns:
        True if exception is retryable

    Examples:
        >>> is_retryable_error(Exception("429 Rate Limit"))
        True

        >>> is_retryable_error(Exception("Request timeout"))
        True

        >>> is_retryable_error(Exception("Invalid API key"))
        False
    """
    return is_rate_limit_error(exception) or is_timeout_error(exception)


# Example usage
if __name__ == "__main__":
    import random

    print("=== Retry Decorator Examples ===\n")

    # Example 1: Flaky function that succeeds on 3rd try
    print("1. Testing flaky function (succeeds on 3rd try)...")

    attempt_counter = {"count": 0}

    @retry_with_exponential_backoff(max_retries=5, base_delay=0.5)
    def flaky_function():
        attempt_counter["count"] += 1
        print(f"   Attempt {attempt_counter['count']}...")
        if attempt_counter["count"] < 3:
            raise Exception("Temporary failure")
        return "Success!"

    try:
        result = flaky_function()
        print(f"   ✓ Result: {result}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Example 2: Rate limit with custom retry condition
    print("\n2. Testing rate limit retry...")

    @retry_with_exponential_backoff(
        max_retries=2,
        base_delay=0.5,
        retry_condition=is_rate_limit_error
    )
    def rate_limited_function():
        if random.random() < 0.7:
            raise Exception("429 Rate limit exceeded")
        return "Success!"

    try:
        result = rate_limited_function()
        print(f"   ✓ Result: {result}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Example 3: Non-retryable error (should fail immediately)
    print("\n3. Testing non-retryable error...")

    @retry_with_exponential_backoff(
        max_retries=3,
        retry_condition=is_retryable_error
    )
    def non_retryable_function():
        raise Exception("Invalid API key")  # Not retryable

    try:
        result = non_retryable_function()
        print(f"   ✓ Result: {result}")
    except Exception as e:
        print(f"   ✓ Correctly failed immediately: {e}")

    # Example 4: Exponential backoff timing
    print("\n4. Testing exponential backoff timing...")

    delays = []

    def track_delay(exception, attempt, delay):
        delays.append(delay)

    @retry_with_exponential_backoff(
        max_retries=4,
        base_delay=1.0,
        backoff_factor=2.0,
        on_retry=track_delay
    )
    def always_fails():
        raise Exception("Always fails")

    try:
        always_fails()
    except:
        print(f"   Delays: {[f'{d:.1f}s' for d in delays]}")
        print(f"   Expected: ['1.0s', '2.0s', '4.0s', '8.0s']")
