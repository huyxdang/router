"""Check that all API endpoints are reachable and keys are valid."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    TTC_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY,
    MODELS, BEAR_API_URL, BEAR_MODEL, EMBEDDING_MODEL, JUDGE_MODEL,
)

TEST_PROMPT = "The quick brown fox jumps over the lazy dog."
PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def check_bear():
    """Test Bear compression API."""
    if not TTC_API_KEY:
        return SKIP, "TTC_API_KEY not set"

    import httpx
    try:
        resp = httpx.post(
            BEAR_API_URL,
            headers={"Authorization": f"Bearer {TTC_API_KEY}", "Content-Type": "application/json"},
            json={"model": BEAR_MODEL, "input": TEST_PROMPT, "compression_settings": {"aggressiveness": 0.4}},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return PASS, f"compressed {data['original_input_tokens']} -> {data['output_tokens']} tokens"
    except Exception as e:
        return FAIL, str(e)


def check_openai():
    """Test OpenAI chat completions."""
    if not OPENAI_API_KEY:
        return SKIP, "OPENAI_API_KEY not set"

    import openai
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_completion_tokens=10,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        return PASS, f'"{text}" ({resp.usage.prompt_tokens}+{resp.usage.completion_tokens} tokens)'
    except Exception as e:
        return FAIL, str(e)


def check_anthropic():
    """Test Anthropic messages API."""
    if not ANTHROPIC_API_KEY:
        return SKIP, "ANTHROPIC_API_KEY not set"

    import anthropic
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        text = resp.content[0].text.strip()
        return PASS, f'"{text}" ({resp.usage.input_tokens}+{resp.usage.output_tokens} tokens)'
    except Exception as e:
        return FAIL, str(e)


def check_openai_embeddings():
    """Test OpenAI embeddings API."""
    if not OPENAI_API_KEY:
        return SKIP, "OPENAI_API_KEY not set"

    import openai
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[TEST_PROMPT])
        dim = len(resp.data[0].embedding)
        return PASS, f"{EMBEDDING_MODEL} -> {dim}-dim vector"
    except Exception as e:
        return FAIL, str(e)


def check_openrouter():
    """Test OpenRouter API."""
    if not OPENROUTER_API_KEY:
        return SKIP, "OPENROUTER_API_KEY not set"

    import openai
    try:
        client = openai.OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
        resp = client.chat.completions.create(
            model="openrouter/auto",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        model_used = resp.model
        return PASS, f'"{text}" via {model_used}'
    except Exception as e:
        return FAIL, str(e)


def check_openai_batch():
    """Test OpenAI batch API access (just list batches, no cost)."""
    if not OPENAI_API_KEY:
        return SKIP, "OPENAI_API_KEY not set"

    import openai
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        batches = client.batches.list(limit=1)
        return PASS, f"batch API accessible (judge model: {JUDGE_MODEL})"
    except Exception as e:
        return FAIL, str(e)


def main():
    checks = [
        ("Bear Compression", check_bear),
        ("OpenAI (chat)", check_openai),
        ("Anthropic (chat)", check_anthropic),
        ("OpenAI (embeddings)", check_openai_embeddings),
        ("OpenAI (batch API)", check_openai_batch),
        ("OpenRouter", check_openrouter),
    ]

    print("=" * 70)
    print("API ENDPOINT CHECK")
    print("=" * 70)

    results = []
    for name, fn in checks:
        print(f"\n  Checking {name}...", end="", flush=True)
        status, detail = fn()
        results.append((name, status, detail))
        print(f" {status}")
        if detail:
            print(f"    {detail}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    skipped = sum(1 for _, s, _ in results if s == SKIP)

    for name, status, detail in results:
        icon = {"PASS": "+", "FAIL": "X", "SKIP": "-"}[status]
        print(f"  [{icon}] {name}: {status}")

    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped")

    if failed:
        print("\n  Fix failed endpoints before running the pipeline.")
        sys.exit(1)
    else:
        print("\n  All endpoints OK. Ready to run.")


if __name__ == "__main__":
    main()
