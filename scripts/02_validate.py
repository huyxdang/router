"""Quick smoke test: run a small subset through the full pipeline to confirm everything works."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MODELS, AGGRESSIVENESS_LEVELS, SYSTEM_PROMPTS, DEFAULT_SYSTEM_PROMPT
from router.compress import compress
from router.llm import call_llm
from router.evaluate import compute_cost
from router.data import load_prompts as _load_prompts

# Smoke test settings
PROMPTS_PER_BENCHMARK = 1
SMOKE_AGG_LEVELS = [AGGRESSIVENESS_LEVELS[0], AGGRESSIVENESS_LEVELS[-1]]


def load_prompts(n: int) -> list[dict]:
    """Load first n prompts from each available benchmark."""
    all_prompts = _load_prompts()
    # Group by benchmark, take first n from each
    by_bench = {}
    for p in all_prompts:
        by_bench.setdefault(p["benchmark"], []).append(p)
    result = []
    for prompts in by_bench.values():
        result.extend(prompts[:n])
    return result


def main():
    print("=" * 70)
    print("SMOKE TEST")
    print("=" * 70)

    prompts = load_prompts(PROMPTS_PER_BENCHMARK)
    if not prompts:
        print("No prompts found. Run scripts/01_prepare_data.py first.")
        return

    print(f"\n  Prompts: {len(prompts)}")
    print(f"  Agg levels: {SMOKE_AGG_LEVELS}")
    print(f"  Models: {[m['name'] for m in MODELS]}")
    total = len(prompts) * len(SMOKE_AGG_LEVELS) * len(MODELS)
    print(f"  Total calls: {total}\n")

    results = []
    for prompt in prompts:
        for agg in SMOKE_AGG_LEVELS:
            print(f"  Compressing {prompt['id']} @ agg={agg}...")
            comp = compress(prompt["text"], agg)
            print(f"    {comp['original_input_tokens']} -> {comp['output_tokens']} tokens "
                  f"({comp['removal_rate']:.1%} removed)")

            for model in MODELS:
                print(f"    Calling {model['name']}...")
                sys_prompt = SYSTEM_PROMPTS.get(prompt["benchmark"], DEFAULT_SYSTEM_PROMPT)
                llm_result = call_llm(model, comp["compressed_text"], sys_prompt)

                cost = compute_cost(
                    model,
                    llm_result["input_tokens"],
                    llm_result["output_tokens"],
                    comp["tokens_removed"],
                )

                results.append({
                    "prompt_id": prompt["id"],
                    "model": model["name"],
                    "agg": agg,
                    "total_cost": cost["total_cost_usd"],
                    "latency": llm_result["latency"],
                    "response_preview": llm_result["response_text"][:80],
                })

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Prompt':<12} {'Model':<16} {'Agg':>4} {'Cost':>8} {'Latency':>7}")
    print("-" * 60)
    for r in results:
        print(f"{r['prompt_id']:<12} {r['model']:<16} {r['agg']:>4.1f} ${r['total_cost']:>7.5f} {r['latency']:>6.2f}s")

    print("\n" + "-" * 60)
    print("Response previews:")
    for r in results:
        print(f"  [{r['prompt_id']} / {r['model']} / agg={r['agg']}]: {r['response_preview']}...")

    print("\nSmoke test complete!")


if __name__ == "__main__":
    main()
