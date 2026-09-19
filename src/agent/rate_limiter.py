from __future__ import annotations

import os
import threading
import time


class RateLimiter:
    """Thread-safe, spaced-slot rate limiter for smoothing request bursts.

    Each :meth:`acquire` call reserves the next evenly spaced time slot and
    sleeps until it arrives. The reservation happens under a short lock, but the
    wait itself happens OUTSIDE the lock, so many parallel callers (e.g. the
    LangGraph ``Send`` fanout workers) each receive a distinct, staggered start
    time and then sleep concurrently rather than all firing at once.

    This replaces the previous fixed ``time.sleep(...)`` staggering in the
    worker node. Instead of hard-coding a per-task delay (which added dead time
    and did not adapt to the number of workers), request starts are spaced by a
    single, shared ``min_interval`` — enough to avoid Groq TPM burst spikes,
    while overlapping the waits so total latency stays low.
    """

    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_allowed = 0.0  # monotonic timestamp of the next free slot

    def acquire(self) -> float:
        """Block until this caller's spaced slot arrives. Returns seconds slept."""
        if self.min_interval <= 0:
            return 0.0

        with self._lock:
            now = time.monotonic()
            target = self._next_allowed if self._next_allowed > now else now
            self._next_allowed = target + self.min_interval

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
