import time

import anthropic
from mistralai import Mistral

from config import ANTHROPIC_API_KEY, MISTRAL_API_KEY

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based on the provided context. "
    "Be concise and give the answer directly."
)

_anthropic_client = None
_mistral_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_mistral_client():
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    return _mistral_client


def call_llm(model_config: dict, prompt_text: str) -> dict:
    """Call an LLM and return response with metadata.

    Returns dict with: response_text, input_tokens, output_tokens, latency.
    """
    provider = model_config["provider"]
    model_id = model_config["id"]

    start = time.time()

    if provider == "anthropic":
        client = _get_anthropic_client()
        response = client.messages.create(
            model=model_id,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt_text}],
        )
        latency = time.time() - start
        return {
            "response_text": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency": latency,
        }

    elif provider == "mistral":
        client = _get_mistral_client()
        response = client.chat.complete(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=256,
        )
        latency = time.time() - start
        return {
            "response_text": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "latency": latency,
        }

    else:
        raise ValueError(f"Unknown provider: {provider}")
