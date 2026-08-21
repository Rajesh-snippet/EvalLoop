"""
Retry/backoff decorator — same pattern used in ResearchFlow's utils layer.
Wraps flaky network calls (Groq API) with exponential backoff + jitter.
"""

from __future__ import annotations

import functools
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    retryable_exceptions: tuple = (Exception,),
):
    """Decorator: retries the wrapped function with exponential backoff + jitter.

    On the final failed attempt, re-raises the original exception so callers
    can distinguish "gave up after retries" from other error paths.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exc = e
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (2**attempt), max_delay)
                    delay += random.uniform(0, delay * 0.25)  # jitter
                    time.sleep(delay)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
