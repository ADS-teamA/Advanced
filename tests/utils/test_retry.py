"""
Unit tests for utils.retry module.

Tests exponential backoff retry decorator.
"""

import pytest
import time
from unittest.mock import Mock, call
from src.utils.retry import retry_with_exponential_backoff


class TestRetryWithExponentialBackoff:
    """Test retry_with_exponential_backoff decorator."""

    def test_successful_first_attempt(self):
        """Test function succeeds on first attempt (no retries)."""
        mock_fn = Mock(return_value="success")

        @retry_with_exponential_backoff(max_retries=3)
        def test_function():
            return mock_fn()

        result = test_function()

        assert result == "success"
        assert mock_fn.call_count == 1

    def test_retry_on_failure(self):
        """Test function retries on failure."""
        mock_fn = Mock(side_effect=[Exception("Fail 1"), Exception("Fail 2"), "success"])

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.1)
        def test_function():
            return mock_fn()

        result = test_function()

        assert result == "success"
        assert mock_fn.call_count == 3

    def test_max_retries_exceeded(self):
        """Test exception raised when max retries exceeded."""
        mock_fn = Mock(side_effect=Exception("Always fails"))

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.1)
        def test_function():
            return mock_fn()

        with pytest.raises(Exception, match="Always fails"):
            test_function()

        # Should try: initial + 3 retries = 4 attempts
        assert mock_fn.call_count == 4

    def test_exponential_backoff_delay(self):
        """Test exponential backoff delays increase correctly."""
        delays = []

        def capture_delay(e, attempt, delay):
            delays.append(delay)

        mock_fn = Mock(
            side_effect=[Exception("Fail 1"), Exception("Fail 2"), Exception("Fail 3"), "success"]
        )

        @retry_with_exponential_backoff(
            max_retries=3, base_delay=1.0, backoff_factor=2.0, on_retry=capture_delay
        )
        def test_function():
            return mock_fn()

        result = test_function()

        assert result == "success"
        # Delays should be: 1.0, 2.0, 4.0
        assert len(delays) == 3
        assert delays[0] == 1.0
        assert delays[1] == 2.0
        assert delays[2] == 4.0

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        delays = []

        def capture_delay(e, attempt, delay):
            delays.append(delay)

        mock_fn = Mock(
            side_effect=[Exception("Fail 1"), Exception("Fail 2"), Exception("Fail 3"), "success"]
        )

        @retry_with_exponential_backoff(
            max_retries=3,
            base_delay=10.0,
            max_delay=15.0,
            backoff_factor=2.0,
            on_retry=capture_delay,
        )
        def test_function():
            return mock_fn()

        result = test_function()

        assert result == "success"
        # Delays should be: 10.0, 15.0 (capped), 15.0 (capped)
        assert len(delays) == 3
        assert delays[0] == 10.0
        assert delays[1] == 15.0  # Capped from 20.0
        assert delays[2] == 15.0  # Capped from 40.0

    def test_specific_exception_types(self):
        """Test retry only on specific exception types."""
        mock_fn = Mock(side_effect=[ValueError("Retryable"), "success"])

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.1, exceptions=(ValueError,))
        def test_function():
            return mock_fn()

        result = test_function()

        assert result == "success"
        assert mock_fn.call_count == 2

    def test_non_retryable_exception(self):
        """Test non-retryable exception not retried."""
        mock_fn = Mock(side_effect=TypeError("Not retryable"))

        @retry_with_exponential_backoff(
            max_retries=3, base_delay=0.1, exceptions=(ValueError,)  # Only retry ValueError
        )
        def test_function():
            return mock_fn()

        with pytest.raises(TypeError, match="Not retryable"):
            test_function()

        # Should only try once (no retries for TypeError)
        assert mock_fn.call_count == 1

    def test_retry_condition(self):
        """Test custom retry condition function."""

        def should_retry(e):
            # Only retry if error message contains "retry"
            return "retry" in str(e).lower()

        mock_fn = Mock(
            side_effect=[
                Exception("Please retry"),  # Will retry
                Exception("Do not retry"),  # Will not retry
            ]
        )

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.1, retry_condition=should_retry)
        def test_function():
            return mock_fn()

        with pytest.raises(Exception, match="Do not retry"):
            test_function()

        # First attempt + 1 retry (then fail on non-retryable)
        assert mock_fn.call_count == 2

    def test_on_retry_callback(self):
        """Test on_retry callback is called on each retry."""
        retry_info = []

        def on_retry_callback(e, attempt, delay):
            retry_info.append({"error": str(e), "attempt": attempt, "delay": delay})

        mock_fn = Mock(side_effect=[Exception("Fail 1"), Exception("Fail 2"), "success"])

        @retry_with_exponential_backoff(max_retries=3, base_delay=1.0, on_retry=on_retry_callback)
        def test_function():
            return mock_fn()

        result = test_function()

        assert result == "success"
        assert len(retry_info) == 2  # 2 retries
        assert retry_info[0]["attempt"] == 1
        assert retry_info[1]["attempt"] == 2

    def test_on_retry_callback_failure(self, caplog):
        """Test on_retry callback failure doesn't stop retries."""

        def failing_callback(e, attempt, delay):
            raise Exception("Callback failed")

        mock_fn = Mock(side_effect=[Exception("Fail 1"), "success"])

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.1, on_retry=failing_callback)
        def test_function():
            return mock_fn()

        # Should still succeed despite callback failure
        with caplog.at_level("ERROR"):
            result = test_function()

        assert result == "success"
        assert "Retry callback failed" in caplog.text

    def test_decorator_preserves_function_metadata(self):
        """Test decorator preserves function name and docstring."""

        @retry_with_exponential_backoff(max_retries=3)
        def my_function():
            """This is my function."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "This is my function."

    def test_actual_delay_timing(self):
        """Test that actual delays occur (integration test)."""
        start_time = time.time()

        mock_fn = Mock(side_effect=[Exception("Fail"), "success"])

        @retry_with_exponential_backoff(
            max_retries=2, base_delay=0.5, backoff_factor=2.0  # 0.5 seconds
        )
        def test_function():
            return mock_fn()

        result = test_function()
        elapsed = time.time() - start_time

        assert result == "success"
        # Should take at least 0.5 seconds (1 retry with 0.5s delay)
        assert elapsed >= 0.5
        # But not too long (allow some overhead)
        assert elapsed < 1.0

    def test_api_key_sanitization_in_errors(self, caplog):
        """Test that API keys are sanitized in retry logs."""
        mock_fn = Mock(side_effect=[Exception("API error: sk-ant-secret123"), "success"])

        @retry_with_exponential_backoff(max_retries=2, base_delay=0.1)
        def test_function():
            return mock_fn()

        with caplog.at_level("WARNING"):
            result = test_function()

        assert result == "success"
        # Verify API key is masked in logs
        assert "sk-ant-***" in caplog.text
        assert "secret123" not in caplog.text

    def test_zero_retries(self):
        """Test with max_retries=0 (no retries, fail immediately)."""
        mock_fn = Mock(side_effect=Exception("Fail"))

        @retry_with_exponential_backoff(max_retries=0, base_delay=0.1)
        def test_function():
            return mock_fn()

        with pytest.raises(Exception, match="Fail"):
            test_function()

        # Should only try once
        assert mock_fn.call_count == 1

    def test_function_with_arguments(self):
        """Test decorated function with arguments."""
        mock_fn = Mock(return_value="success")

        @retry_with_exponential_backoff(max_retries=3)
        def test_function(a, b, c=None):
            return mock_fn(a, b, c)

        result = test_function(1, 2, c=3)

        assert result == "success"
        mock_fn.assert_called_once_with(1, 2, 3)

    def test_function_with_kwargs(self):
        """Test decorated function with keyword arguments."""
        mock_fn = Mock(return_value="success")

        @retry_with_exponential_backoff(max_retries=3)
        def test_function(**kwargs):
            return mock_fn(**kwargs)

        result = test_function(x=1, y=2, z=3)

        assert result == "success"
        mock_fn.assert_called_once_with(x=1, y=2, z=3)


# Integration tests
class TestRetryIntegration:
    """Integration tests for retry decorator."""

    def test_realistic_rate_limit_scenario(self):
        """Test realistic rate limit scenario with retry."""
        attempt_count = 0

        def api_call_with_rate_limit():
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count < 3:
                raise Exception("Rate limit exceeded (429)")
            return {"status": "success"}

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.1, exceptions=(Exception,))
        def call_api():
            return api_call_with_rate_limit()

        result = call_api()

        assert result == {"status": "success"}
        assert attempt_count == 3

    def test_immediate_success_no_delay(self):
        """Test immediate success causes no delay."""
        start_time = time.time()

        @retry_with_exponential_backoff(max_retries=3, base_delay=10.0)
        def fast_function():
            return "immediate"

        result = fast_function()
        elapsed = time.time() - start_time

        assert result == "immediate"
        # Should be very fast (no retries, no delays)
        assert elapsed < 0.1
