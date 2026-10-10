from __future__ import annotations

import os
import threading
import time


class RateLimiter:
    """Thread-safe, spaced-slot rate limiter with per-key tracking.

    If key_index is provided (e.g. for multi-account key rotation), callers with
    different keys do NOT block each other: each key runs on its own independent slot schedule.
    Callers on the same key are spaced by min_interval to avoid TPM burst spikes.
    """

    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_allowed_by_key: dict[int, float] = {}
        self._next_allowed_default = 0.0

    def acquire(self, key_index: Optional[int] = None) -> float:
        """Block until this caller's spaced slot arrives. Returns seconds slept."""
        if self.min_interval <= 0:
            return 0.0

        with self._lock:
            now = time.monotonic()
            if key_index is not None:
                current_next = self._next_allowed_by_key.get(key_index, 0.0)
                target = current_next if current_next > now else now
                self._next_allowed_by_key[key_index] = target + self.min_interval
            else:
                target = self._next_allowed_default if self._next_allowed_default > now else now
                self._next_allowed_default = target + self.min_interval

        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)
            return delay
        return 0.0


def _default_interval() -> float:
    """Minimum spacing (seconds) between Groq requests; override via env."""
    try:
        return max(0.0, float(os.getenv("GROQ_MIN_REQUEST_INTERVAL", "1.0")))
    except (TypeError, ValueError):
        return 1.0


# Shared limiter used around parallel Groq LLM calls. Parallel workers import
# this single instance so their request starts are spaced against each other.
groq_rate_limiter = RateLimiter(_default_interval())
