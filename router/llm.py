from __future__ import annotations

import time

import anthropic
import openai

from config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL, OPENROUTER_MODEL_IDS,
    LLM_MAX_TOKENS, LLM_TEMPERATURE,
    DEFAULT_SYSTEM_PROMPT,
)

# Sync clients
_anthropic_client = None
_openai_client = None
_openrouter_client = None

# Async clients
_async_anthropic_client = None
_async_openai_client = None
_async_openrouter_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_async_anthropic_client():
    global _async_anthropic_client
    if _async_anthropic_client is None:
        _async_anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _async_anthropic_client


def _get_async_openai_client():
    global _async_openai_client
    if _async_openai_client is None:
        _async_openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _async_openai_client


def _get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = openai.OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    return _openrouter_client


def _get_async_openrouter_client():
    global _async_openrouter_client
    if _async_openrouter_client is None:
        _async_openrouter_client = openai.AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    return _async_openrouter_client


def call_llm(model_config: dict, prompt_text: str, system_prompt: str | None = None) -> dict:
    """Call an LLM and return response with metadata.

    Returns dict with: response_text, input_tokens, output_tokens, latency.
    """
    provider = model_config["provider"]
    model_id = model_config["id"]
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    start = time.time()

    if provider == "anthropic":
        client = _get_anthropic_client()
        response = client.messages.create(
            model=model_id,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt_text}],
        )
        latency = time.time() - start
        text = response.content[0].text if response.content else ""
        return {
            "response_text": text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency": latency,
        }

    elif provider == "openai":
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt_text},
            ],
            max_completion_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        latency = time.time() - start
        return {
            "response_text": response.choices[0].message.content or "",
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "latency": latency,
        }

    else:
        raise ValueError(f"Unknown provider: {provider}")


async def call_llm_async(model_config: dict, prompt_text: str, system_prompt: str | None = None) -> dict:
    """Async version of call_llm."""
    provider = model_config["provider"]
    model_id = model_config["id"]
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    start = time.time()

    if provider == "anthropic":
        client = _get_async_anthropic_client()
        response = await client.messages.create(
            model=model_id,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt_text}],
        )
        latency = time.time() - start
        return {
            "response_text": response.content[0].text if response.content else "",
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency": latency,
        }

    elif provider == "openai":
        client = _get_async_openai_client()
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt_text},
            ],
            max_completion_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        latency = time.time() - start
        return {
            "response_text": response.choices[0].message.content or "",
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "latency": latency,
        }

    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_openrouter(prompt_text: str, allowed_models: list[str] | None = None,
                    system_prompt: str | None = None) -> dict:
    """Call OpenRouter auto-router, optionally restricted to specific models.

    Args:
        prompt_text: The prompt to send.
        allowed_models: List of our model IDs (e.g. "gpt-4o-mini") to restrict to.
                       If None, OpenRouter picks from all available models.
        system_prompt: Override system prompt. Falls back to DEFAULT_SYSTEM_PROMPT.

    Returns dict with: response_text, input_tokens, output_tokens, latency, model_used.
    """
    client = _get_openrouter_client()
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    # Map our model IDs to OpenRouter format
    or_models = None
    if allowed_models:
        or_models = [OPENROUTER_MODEL_IDS[m] for m in allowed_models
                     if m in OPENROUTER_MODEL_IDS]

    start = time.time()
    extra_body = {}
    if or_models:
        extra_body["models"] = or_models

    response = client.chat.completions.create(
        model="openrouter/auto",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt_text},
        ],
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        extra_body=extra_body,
    )
    latency = time.time() - start

    return {
        "response_text": response.choices[0].message.content or "",
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "latency": latency,
        "model_used": response.model,
    }


async def call_openrouter_async(prompt_text: str, allowed_models: list[str] | None = None,
                               system_prompt: str | None = None) -> dict:
    """Async version of call_openrouter."""
    client = _get_async_openrouter_client()
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    or_models = None
    if allowed_models:
        or_models = [OPENROUTER_MODEL_IDS[m] for m in allowed_models
                     if m in OPENROUTER_MODEL_IDS]

    start = time.time()
    extra_body = {}
    if or_models:
        extra_body["models"] = or_models

    response = await client.chat.completions.create(
        model="openrouter/auto",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt_text},
        ],
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        extra_body=extra_body,
    )
    latency = time.time() - start

    return {
        "response_text": response.choices[0].message.content or "",
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "latency": latency,
        "model_used": response.model,
    }
