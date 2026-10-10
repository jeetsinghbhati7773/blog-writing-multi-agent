from __future__ import annotations

import itertools
import logging
import os
from typing import List, Optional

from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _clean_key(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = str(val).strip().strip('"').strip("'")
    if "#" in s:
        s = s.split("#")[0].strip()
    return s if s else None


def _clean_model(model: Optional[str]) -> str:
    if not model:
        return "qwen/qwen3.8-27b"
    m = str(model).strip().strip('"').strip("'")
    decommissioned = {
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
        "llama-3.1-8b-instant",
    }
    if m in decommissioned or not m:
        return "qwen/qwen3.8-27b"
    return m


import threading

class MultiAccountGroqManager:
    """
    Manages multi-account / multi-key Groq API rotation using Round-Robin distribution
    with automatic Runnable fallback chains (`with_fallbacks`) to prevent HTTP 429 rate limit delays.
    Supports index-based allocation so parallel worker nodes are pinned to distinct accounts.
    """

    def __init__(self, api_keys: Optional[List[str]] = None, default_model: Optional[str] = None):
        raw_model = default_model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.default_model = _clean_model(raw_model)
        self._lock = threading.Lock()

        if not api_keys:
            raw_keys = []
            for i in range(1, 11):
                raw_keys.append(os.getenv(f"GROQ_API_KEY_ACCOUNT_{i}"))
                raw_keys.append(os.getenv(f"GROQ_API_KEY_{i}"))
            api_keys = [_clean_key(k) for k in raw_keys if _clean_key(k)]

        if not api_keys:
            try:
                import streamlit as st
                for i in range(1, 11):
                    for prefix in [f"GROQ_API_KEY_ACCOUNT_{i}", f"GROQ_API_KEY_{i}"]:
                        cleaned = _clean_key(st.secrets.get(prefix))
                        if cleaned:
                            api_keys.append(cleaned)
            except Exception as e:
                logger.debug("Streamlit secrets key load skipped: %s", e)

        if not api_keys:
            single_key = _clean_key(os.getenv("GROQ_API_KEY"))
            if not single_key:
                try:
                    import streamlit as st
                    single_key = _clean_key(st.secrets.get("GROQ_API_KEY"))
                except Exception:
                    pass
            if single_key:
                api_keys = [single_key]

        # Deduplicate keys while maintaining order
        cleaned_unique_keys = list(dict.fromkeys(api_keys)) if api_keys else []
        self.api_keys = cleaned_unique_keys
        self._pool_cycle = itertools.cycle(range(len(self.api_keys))) if self.api_keys else None

    def get_key_count(self) -> int:
        """Returns the number of unique active API keys loaded in the pool."""
        return len(self.api_keys)

    def get_llm_pool(self, model_name: Optional[str] = None, temperature: float = 0.7) -> List[ChatGroq]:
        """
        Creates a list of ChatGroq instances for each configured API key.
        """
        model = model_name or self.default_model
        if not self.api_keys:
            groq_key = os.getenv("GROQ_API_KEY", "")
            return [
                ChatGroq(
                    model=model,
                    temperature=temperature,
                    groq_api_key=groq_key,
                    max_retries=1,
                )
            ]

        # Fast failover so with_fallbacks triggers instantly
        return [
            ChatGroq(
                model=model,
                temperature=temperature,
                groq_api_key=key,
                max_retries=1,
            )
            for key in self.api_keys
        ]

    def get_runnable(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        index: Optional[int] = None,
    ) -> Runnable:
        """
        Returns a selected ChatGroq instance backed by an automatic fallback chain
        (`with_fallbacks`) across the remaining account keys upon 429/error.
        If `index` is specified (e.g. from parallel section workers), pins primary key to (index % num_keys).
        Otherwise rotates round-robin thread-safely.
        """
        pool = self.get_llm_pool(model_name=model_name, temperature=temperature)
        if len(pool) <= 1:
            return pool[0]

        if index is not None:
            primary_idx = int(index) % len(pool)
        else:
            with self._lock:
                primary_idx = next(self._pool_cycle) if self._pool_cycle else 0
                primary_idx = primary_idx % len(pool)

        primary_llm = pool[primary_idx]

        fallback_llms = [
            pool[(primary_idx + offset) % len(pool)]
            for offset in range(1, len(pool))
        ]

        return primary_llm.with_fallbacks(fallbacks=fallback_llms)


# Backward compatibility alias
GroqKeyRotator = MultiAccountGroqManager

# Global Manager Instance
groq_manager = MultiAccountGroqManager()
