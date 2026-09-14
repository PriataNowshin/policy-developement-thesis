"""OpenAI-compatible LLM client utilities.

This module uses the official openai Python SDK.
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv
from openai import (
    OpenAI, 
    APIConnectionError, 
    AuthenticationError, 
    APITimeoutError, 
    RateLimitError, 
    InternalServerError,
)

from .constants import (
    BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF_SEC,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SEC,
)
from .prompts import (
    REQUIREMENT_GENERATION_SYSTEM_PROMPT,
    REQUIREMENT_GENERATION_USER_PROMPT,
    FUNCTION_GENERATION_SYSTEM_PROMPT,
    FUNCTION_GENERATION_USER_PROMPT
)

load_dotenv(override=True)

RETRYABLE_ERRORS = (APIConnectionError, AuthenticationError, APITimeoutError, RateLimitError, InternalServerError)


class InvalidLLMResponseError(RuntimeError):
    """Raised when a provider response has no usable completion choice."""

@dataclass
class LLMConfig:
    """Holds LLM API configuration."""
    api_key: str
    model: str
    base_url: str
    temperature: float
    timeout_sec: int
    max_retries: int
    retry_backoff_sec: float


def build_config(model: str) -> LLMConfig:
    """Build an LLM_Config from the environment and the chosen model.

    This reads the API key from the API_KEY environment variable and combines it
    with defaults from src.constants.

    Args:
        model: Model identifier to send to the OpenAI-compatible API.

    Returns:
        A populated LLM_Config instance.

    Raises:
        RuntimeError: If API_KEY is missing from the environment.
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_KEY")

    return LLMConfig(
        api_key=api_key,
        model=model,
        base_url=BASE_URL,
        temperature=DEFAULT_TEMPERATURE,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
        max_retries=DEFAULT_MAX_RETRIES,
        retry_backoff_sec=DEFAULT_RETRY_BACKOFF_SEC,
    )


def invoke_llm(messages: List[Dict[str, str]], config: LLMConfig) -> str:
    """Invoke an OpenAI-compatible chat-completions endpoint with retries.

    Args:
        messages: Chat messages in OpenAI format, e.g.
            [{"role": "system", "content": "..."}, ...].
        config: API and retry configuration.

    Returns:
        The first choice's message content as a stripped string.

    Raises:
        RuntimeError: If the request fails after exhausting retries.
    """
    client = OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
    )

    for attempt in range(config.max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                timeout=config.timeout_sec,
            )

            choices = completion.choices
            if not choices or choices[0].message is None:
                raise InvalidLLMResponseError(
                    "Provider returned no completion choices."
                )

            content = choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise InvalidLLMResponseError(
                    "Provider returned an empty completion."
                )

            return content.strip()

        except RETRYABLE_ERRORS + (InvalidLLMResponseError,) as exc:
            if attempt >= config.max_retries:
                raise RuntimeError(f"LLM request failed: {exc}") from exc
            sleep_for = config.retry_backoff_sec * (2 ** attempt)
            time.sleep(sleep_for)

    raise RuntimeError("LLM request failed after retries.")


def generate_requirement(old_code: str, new_code: str, config: LLMConfig) -> str:
    """Generate a functional requirement describing a code change.

    Args:
        old_code: The original (pre-change) function code.
        new_code: The updated (post-change) function code.
        config: API and retry configuration.

    Returns:
        A natural-language requirement describing what changed.
    """
    messages = [
        {"role": "system", "content": REQUIREMENT_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": REQUIREMENT_GENERATION_USER_PROMPT.format(old_code=old_code, new_code=new_code)},
    ]

    return invoke_llm(messages, config)


def generate_function(old_code: str, requirement: str, config: LLMConfig) -> str:
    """Regenerate updated function code given an old version and a requirement.

    Args:
        old_code: The original (pre-change) function code.
        requirement: Natural-language requirement describing the desired change.
        config: API and retry configuration.

    Returns:
        The regenerated function code as a string.
    """
    messages = [
        {"role": "system", "content": FUNCTION_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": FUNCTION_GENERATION_USER_PROMPT.format(old_code=old_code, requirement=requirement)},
    ]

    return invoke_llm(messages, config)
