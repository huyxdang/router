"""Grid search: run every (prompt, aggressiveness, model) combination.

Supports auto-resume — if interrupted, re-run and it picks up where it left off.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MODELS, AGGRESSIVENESS_LEVELS
from router.compress import compress
from router.llm import call_llm
from router.evaluate import exact_match, f1_score, contains_answer, compute_cost

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

RESULTS_PATH = os.path.join(RESULTS_DIR, "grid_results.parquet")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "grid_results_checkpoint.parquet")
COMPRESS_CACHE_PATH = os.path.join(RESULTS_DIR, "compressed_cache.json")

TRIAL_ID = "run_001"


def load_prompts() -> list[dict]:
    """Load all benchmark prompts."""
    prompts = []
    for filename in ["squad2_subset.json", "finqa_subset.json"]:
        path = os.path.join(DATA_DIR, filename)
        with open(path) as f:
            prompts.extend(json.load(f))
    return prompts


def load_existing_results() -> tuple[list[dict], set]:
    """Load existing results and return (records, completed_keys)."""
    for path in [CHECKPOINT_PATH, RESULTS_PATH]:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            records = df.to_dict("records")
            completed = {
                (r["prompt_id"], r["aggressiveness"], r["model_name"])
                for r in records
            }
            print(f"  Resumed from {path}: {len(records)} existing records")
            return records, completed
    return [], set()


def load_compress_cache() -> dict:
    """Load cached compression results."""
    if os.path.exists(COMPRESS_CACHE_PATH):
        with open(COMPRESS_CACHE_PATH) as f:
            cache = json.load(f)
        print(f"  Loaded compression cache: {len(cache)} entries")
        return cache
    return {}


def save_compress_cache(cache: dict):
    with open(COMPRESS_CACHE_PATH, "w") as f:
        json.dump(cache, f)


def save_checkpoint(records: list[dict]):
    df = pd.DataFrame(records)
    df.to_parquet(CHECKPOINT_PATH, index=False)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 80)
    print("GRID SEARCH")
    print("=" * 80)

    # Load prompts
    prompts = load_prompts()
    print(f"\nPrompts: {len(prompts)}")
    print(f"Aggressiveness levels: {AGGRESSIVENESS_LEVELS}")
    print(f"Models: {[m['name'] for m in MODELS]}")

    total_combos = len(prompts) * len(AGGRESSIVENESS_LEVELS) * len(MODELS)
    print(f"Total combinations: {total_combos}")

    # Load existing state for resume
    print("\nChecking for existing progress...")
    records, completed = load_existing_results()
    compress_cache = load_compress_cache()

    remaining = total_combos - len(completed)
    if remaining == 0:
        print("\nAll combinations already completed!")
        return
    print(f"  Remaining: {remaining} / {total_combos}")

    # Phase 1: Compress all prompts at all aggressiveness levels
    print("\n" + "-" * 80)
    print("PHASE 1: Compression")
    print("-" * 80)

    compress_total = len(prompts) * len(AGGRESSIVENESS_LEVELS)
    compress_done = len(compress_cache)
    print(f"  {compress_done}/{compress_total} already cached")

    for i, prompt in enumerate(prompts):
        for agg in AGGRESSIVENESS_LEVELS:
            cache_key = f"{prompt['id']}_agg{agg}"
            if cache_key in compress_cache:
                continue

            try:
                result = compress(prompt["text"], agg)
                compress_cache[cache_key] = {
                    "compressed_text": result["compressed_text"],
                    "original_input_tokens": result["original_input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "compression_ratio": result["compression_ratio"],
                    "tokens_removed": result["tokens_removed"],
                    "removal_rate": result["removal_rate"],
                }
                compress_done += 1
            except Exception as e:
                print(f"  ERROR compressing {prompt['id']} @ agg={agg}: {e}")
                continue

        # Save cache every 10 prompts
        if (i + 1) % 10 == 0:
            save_compress_cache(compress_cache)
            print(f"  Compressed {compress_done}/{compress_total} "
                  f"(prompt {i + 1}/{len(prompts)})")

    save_compress_cache(compress_cache)
    print(f"  Compression complete: {compress_done}/{compress_total}")

    # Phase 2: LLM calls
    print("\n" + "-" * 80)
    print("PHASE 2: LLM Calls")
    print("-" * 80)

    calls_made = 0
    errors = 0

    for i, prompt in enumerate(prompts):
        for agg in AGGRESSIVENESS_LEVELS:
            cache_key = f"{prompt['id']}_agg{agg}"
            if cache_key not in compress_cache:
                continue
            comp = compress_cache[cache_key]

            for model in MODELS:
                combo_key = (prompt["id"], agg, model["name"])
                if combo_key in completed:
                    continue

                try:
                    llm_result = call_llm(model, comp["compressed_text"])

                    # Evaluate
                    em = exact_match(llm_result["response_text"], prompt["ground_truth"])
                    f1 = f1_score(llm_result["response_text"], prompt["ground_truth"])
                    ca = contains_answer(llm_result["response_text"], prompt["ground_truth"])

                    # Cost
                    cost = compute_cost(
                        model,
                        llm_result["input_tokens"],
                        llm_result["output_tokens"],
                        comp["tokens_removed"],
                    )

                    record = {
                        "prompt_id": prompt["id"],
                        "benchmark": prompt["benchmark"],
                        "aggressiveness": agg,
                        "model_name": model["name"],
                        "model_id": model["id"],
                        "model_provider": model["provider"],
                        "model_cost_per_1m_input": model["cost_per_1m_input"],
                        "model_cost_per_1m_output": model["cost_per_1m_output"],
                        "original_input_tokens": comp["original_input_tokens"],
                        "output_tokens": comp["output_tokens"],
                        "compression_ratio": comp["compression_ratio"],
                        "tokens_removed": comp["tokens_removed"],
                        "removal_rate": comp["removal_rate"],
                        "llm_response": llm_result["response_text"],
                        "llm_input_tokens": llm_result["input_tokens"],
                        "llm_output_tokens": llm_result["output_tokens"],
                        "correct": em,
                        "f1_score": f1,
                        "contains_answer": ca,
                        "input_cost_usd": cost["input_cost_usd"],
                        "output_cost_usd": cost["output_cost_usd"],
                        "total_llm_cost_usd": cost["total_llm_cost_usd"],
                        "bear_cost_usd": cost["bear_cost_usd"],
                        "total_cost_usd": cost["total_cost_usd"],
                        "latency_seconds": llm_result["latency"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trial_id": TRIAL_ID,
                    }

                    records.append(record)
                    completed.add(combo_key)
                    calls_made += 1

                except Exception as e:
                    print(f"  ERROR {prompt['id']} / {model['name']} / agg={agg}: {e}")
                    errors += 1
                    continue

        # Checkpoint every 5 prompts
        if (i + 1) % 5 == 0:
            save_checkpoint(records)
            done = len(completed)
            print(f"  Progress: {done}/{total_combos} "
                  f"({done/total_combos:.1%}) | "
                  f"prompt {i + 1}/{len(prompts)} | "
                  f"+{calls_made} calls this run | "
                  f"{errors} errors")

    # Final save
    save_checkpoint(records)
    df = pd.DataFrame(records)
    df.to_parquet(RESULTS_PATH, index=False)

    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)
    print(f"  Total records: {len(records)}")
    print(f"  New calls this run: {calls_made}")
    print(f"  Errors: {errors}")
    print(f"  Saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
