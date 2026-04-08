"""
Centralized LLM client for all API calls across the codebase.

All LLM calls should go through this module instead of creating
OpenAI clients directly. Uses Qwen 3.5 Plus via DashScope API.
"""

import os
import random
import threading
import time
from typing import Optional

from openai import OpenAI


def _load_env_file():
    """Load .env file from project root into os.environ (no external deps)."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file()

# Model configuration
LLM_MODEL = "qwen3.5-plus"
LLM_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "90"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_MAX_IN_FLIGHT = int(os.getenv("LLM_MAX_IN_FLIGHT", "4"))

_client: Optional[OpenAI] = None
_sema = threading.Semaphore(max(1, LLM_MAX_IN_FLIGHT))


def get_client() -> OpenAI:
    """Return a singleton OpenAI-compatible client for DashScope."""
    global _client
    if _client is None:
        if not LLM_API_KEY:
            raise ValueError(
                "DASHSCOPE_API_KEY not set in environment or .env file"
            )
        _client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=LLM_TIMEOUT_S,
        )
    return _client


def chat_completion(
    messages: list[dict],
    model: str = LLM_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    **kwargs,
) -> str:
    """
    Make a chat completion call and return the response content.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: Model name (defaults to qwen3.5-plus).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        **kwargs: Additional arguments passed to the API call.

    Returns:
        The content string from the first choice.
    """
    client = get_client()

    # DashScope can be much slower per-request than OpenAI; parallelism helps,
    # but too much concurrency can trigger throttling. Gate in-flight calls.
    with _sema:
        last_err: Optional[Exception] = None
        for attempt in range(max(1, LLM_MAX_RETRIES) + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                if attempt >= max(1, LLM_MAX_RETRIES):
                    break
                # Exponential backoff + jitter
                sleep_s = min(8.0, (0.6 * (2**attempt)) + random.random() * 0.4)
                time.sleep(sleep_s)

        raise last_err if last_err else RuntimeError("LLM call failed without exception")


def get_camel_model():
    """
    Create a CAMEL ModelFactory model using Qwen 3.5 Plus via DashScope.

    Returns a model backend compatible with CAMEL's agent framework.
    """
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=LLM_MODEL,
        url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )
