import httpx

from config import TTC_API_KEY, BEAR_API_URL, BEAR_MODEL


def compress(text: str, aggressiveness: float) -> dict:
    """Compress text using the TTC bear API.

    Returns dict with: compressed_text, original_input_tokens, output_tokens,
    compression_ratio, tokens_removed, removal_rate.
    """
    if aggressiveness == 0.0:
        # No compression — estimate token count as words * 1.3
        approx_tokens = int(len(text.split()) * 1.3)
        return {
            "compressed_text": text,
            "original_input_tokens": approx_tokens,
            "output_tokens": approx_tokens,
            "compression_ratio": 1.0,
            "tokens_removed": 0,
            "removal_rate": 0.0,
        }

    response = httpx.post(
        BEAR_API_URL,
        headers={
            "Authorization": f"Bearer {TTC_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": BEAR_MODEL,
            "input": text,
            "compression_settings": {
                "aggressiveness": aggressiveness,
            },
        },
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()

    original_tokens = data["original_input_tokens"]
    output_tokens = data["output_tokens"]
    tokens_removed = original_tokens - output_tokens

    return {
        "compressed_text": data["output"],
        "original_input_tokens": original_tokens,
        "output_tokens": output_tokens,
        "compression_ratio": output_tokens / original_tokens if original_tokens > 0 else 1.0,
        "tokens_removed": tokens_removed,
        "removal_rate": tokens_removed / original_tokens if original_tokens > 0 else 0.0,
    }
