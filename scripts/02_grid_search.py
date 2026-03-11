"""Grid search: run every (prompt, aggressiveness, model) combination.

Supports:
- Async batching (concurrent LLM calls)
- Auto-resume from checkpoint
- Retry pass for failed combos
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from config import (
    MODELS, AGGRESSIVENESS_LEVELS, BENCHMARKS,
    DATA_DIR, RESULTS_DIR, BATCH_SIZE, CHECKPOINT_EVERY,
    SYSTEM_PROMPTS, DEFAULT_SYSTEM_PROMPT,
    TRAIN_FRACTION, VAL_FRACTION, RANDOM_SEED,
)
from router.compress import compress, compress_async
from router.llm import call_llm_async
from router.evaluate import compute_cost

RESULTS_PATH = os.path.join(str(RESULTS_DIR), "grid_results.parquet")
CHECKPOINT_PATH = os.path.join(str(RESULTS_DIR), "grid_results_checkpoint.parquet")
COMPRESS_CACHE_PATH = os.path.join(str(RESULTS_DIR), "compressed_cache.json")

TRIAL_ID = "run_001"
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds
RATE_LIMIT_DELAY = 30  # seconds to wait on 429


def load_prompts() -> list[dict]:
    """Load val+test prompts only (train prompts don't need LLM calls)."""
    from router.data import load_prompts as _load
    all_prompts = _load()

    # Same split logic as 03_build_router.py — deterministic by seed
    prompt_ids = [p["id"] for p in all_prompts]
    rng = np.random.RandomState(RANDOM_SEED)
    ids = prompt_ids.copy()
    rng.shuffle(ids)

    n_train = int(len(ids) * TRAIN_FRACTION)
    val_test_ids = set(ids[n_train:])  # everything after train

    prompts = [p for p in all_prompts if p["id"] in val_test_ids]
    print(f"  Filtered to val+test: {len(prompts)} / {len(all_prompts)} prompts "
          f"(skipping {len(all_prompts) - len(prompts)} train prompts)")
    return prompts


def load_existing_results() -> tuple[list[dict], set]:
    """Load existing results. Picks whichever file has more records."""
    best_records = []
    best_path = None

    for path in [CHECKPOINT_PATH, RESULTS_PATH]:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if len(df) > len(best_records):
                best_records = df.to_dict("records")
                best_path = path

    if best_records:
        completed = {
            (r["prompt_id"], r["aggressiveness"], r["model_name"])
            for r in best_records
        }
        print(f"  Resumed from {best_path}: {len(best_records)} existing records")
        return best_records, completed

    return [], set()


def load_compress_cache() -> dict:
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
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(CHECKPOINT_PATH, index=False)


def build_record(prompt, agg, comp, model, llm_result):
    """Build a result record from all components."""
    cost = compute_cost(
        model,
        llm_result["input_tokens"],
        llm_result["output_tokens"],
        comp["tokens_removed"],
    )

    return {
        "prompt_id": prompt["id"],
        "benchmark": prompt["benchmark"],
        "aggressiveness": agg,
        "model_name": model["name"],
        "model_id": model["id"],
        "model_provider": model["provider"],
        "model_cost_per_1m_input": model["cost_per_1m_input"],
        "model_cost_per_1m_output": model["cost_per_1m_output"],
        "original_input_tokens": comp["original_input_tokens"],
        "compressed_tokens": comp["output_tokens"],
        "compression_ratio": comp["compression_ratio"],
        "tokens_removed": comp["tokens_removed"],
        "removal_rate": comp["removal_rate"],
        "llm_response": llm_result["response_text"],
        "llm_input_tokens": llm_result["input_tokens"],
        "llm_output_tokens": llm_result["output_tokens"],
        "input_cost_usd": cost["input_cost_usd"],
        "output_cost_usd": cost["output_cost_usd"],
        "total_llm_cost_usd": cost["total_llm_cost_usd"],
        "bear_cost_usd": cost["bear_cost_usd"],
        "total_cost_usd": cost["total_cost_usd"],
        "latency_seconds": llm_result["latency"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trial_id": TRIAL_ID,
    }


def _is_rate_limit(e: Exception) -> bool:
    """Check if an exception is a rate limit error."""
    msg = str(e).lower()
    if "429" in msg or "rate" in msg:
        return True
    if hasattr(e, "status_code") and e.status_code == 429:
        return True
    return False


def _get_retry_after(e: Exception) -> float:
    """Extract retry-after seconds from error, or return default."""
    if hasattr(e, "response") and hasattr(e.response, "headers"):
        ra = e.response.headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
    return RATE_LIMIT_DELAY


async def process_one(prompt, agg, comp, model):
    """Process a single (prompt, agg, model) combo. Returns (combo_key, record) or (combo_key, None) on failure."""
    combo_key = (prompt["id"], agg, model["name"])
    sys_prompt = SYSTEM_PROMPTS.get(prompt["benchmark"], DEFAULT_SYSTEM_PROMPT)

    for attempt in range(MAX_RETRIES + 1):
        try:
            llm_result = await call_llm_async(model, comp["compressed_text"], sys_prompt)
            record = build_record(prompt, agg, comp, model, llm_result)
            return combo_key, record
        except Exception as e:
            if attempt < MAX_RETRIES:
                if _is_rate_limit(e):
                    delay = _get_retry_after(e)
                    print(f"  RATE LIMITED {model['name']} — waiting {delay:.0f}s "
                          f"(attempt {attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(delay)
                else:
                    print(f"  RETRY {attempt + 1}/{MAX_RETRIES} "
                          f"{prompt['id']} / {model['name']} / agg={agg}: {e}")
                    await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                print(f"  FAILED {prompt['id']} / {model['name']} / agg={agg}: {e}")
                return combo_key, None


async def run_batch(tasks):
    """Run a batch of tasks concurrently."""
    return await asyncio.gather(*tasks)


async def main():
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
    print(f"Batch size: {BATCH_SIZE} concurrent calls")

    # Load existing state
    print("\nChecking for existing progress...")
    records, completed = load_existing_results()
    compress_cache = load_compress_cache()

    remaining = total_combos - len(completed)
    if remaining == 0:
        print("\nAll combinations already completed!")
        return
    print(f"  Remaining: {remaining} / {total_combos}")

    # Phase 1: Compress all prompts (async, 3 concurrent to avoid 429)
    COMPRESS_BATCH = 3
    print("\n" + "-" * 80)
    print(f"PHASE 1: Compression (async, batch_size={COMPRESS_BATCH})")
    print("-" * 80)

    compress_total = len(prompts) * len(AGGRESSIVENESS_LEVELS)
    compress_done = len(compress_cache)
    print(f"  {compress_done}/{compress_total} already cached")

    # Build list of pending compressions
    pending_compress = []
    for prompt in prompts:
        for agg in AGGRESSIVENESS_LEVELS:
            cache_key = f"{prompt['id']}_agg{agg}"
            if cache_key not in compress_cache:
                pending_compress.append((prompt, agg, cache_key))

    print(f"  Pending: {len(pending_compress)} compressions")

    async def _compress_one(prompt, agg, cache_key):
        for attempt in range(MAX_RETRIES + 1):
            try:
                result = await compress_async(prompt["text"], agg)
                return cache_key, result, None
            except Exception as e:
                if "429" in str(e) and attempt < MAX_RETRIES:
                    await asyncio.sleep(RATE_LIMIT_DELAY * (attempt + 1))
                elif attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    return cache_key, None, (prompt, agg, str(e))

    compress_failures = []
    for batch_start in range(0, len(pending_compress), COMPRESS_BATCH):
        batch = pending_compress[batch_start:batch_start + COMPRESS_BATCH]
        tasks = [_compress_one(p, a, k) for p, a, k in batch]
        results = await asyncio.gather(*tasks)

        for cache_key, result, error in results:
            if result is not None:
                compress_cache[cache_key] = {
                    "compressed_text": result["compressed_text"],
                    "original_input_tokens": result["original_input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "compression_ratio": result["compression_ratio"],
                    "tokens_removed": result["tokens_removed"],
                    "removal_rate": result["removal_rate"],
                }
                compress_done += 1
            else:
                prompt, agg, err_msg = error
                print(f"  ERROR {prompt['id']} @ agg={agg}: {err_msg}")
                compress_failures.append((prompt, agg))

        if compress_done % 50 == 0 or batch_start + COMPRESS_BATCH >= len(pending_compress):
            save_compress_cache(compress_cache)
            print(f"  Compressed {compress_done}/{compress_total}")

    # Retry failed compressions (sequential — these already failed once)
    if compress_failures:
        print(f"  Retrying {len(compress_failures)} failed compressions...")
        still_failed = 0
        for prompt, agg in compress_failures:
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
                print(f"  RECOVERED {prompt['id']} @ agg={agg}")
            except Exception as e:
                print(f"  STILL FAILED {prompt['id']} @ agg={agg}: {e}")
                still_failed += 1
        if still_failed:
            print(f"  WARNING: {still_failed} compressions failed — "
                  f"those combos will be skipped in Phase 2")

    save_compress_cache(compress_cache)
    print(f"  Compression complete: {compress_done}/{compress_total}")

    # Phase 2: LLM calls (async batched)
    print("\n" + "-" * 80)
    print("PHASE 2: LLM Calls (async, batch_size={})".format(BATCH_SIZE))
    print("-" * 80)

    # Build list of all pending tasks
    pending = []
    for prompt in prompts:
        for agg in AGGRESSIVENESS_LEVELS:
            cache_key = f"{prompt['id']}_agg{agg}"
            if cache_key not in compress_cache:
                continue
            comp = compress_cache[cache_key]

            for model in MODELS:
                combo_key = (prompt["id"], agg, model["name"])
                if combo_key in completed:
                    continue
                pending.append((prompt, agg, comp, model))

    print(f"  Pending: {len(pending)} calls")

    calls_made = 0
    errors = 0
    failed = []  # collect failures for retry pass
    since_checkpoint = 0

    # Process in batches
    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start:batch_start + BATCH_SIZE]
        tasks = [process_one(p, a, c, m) for p, a, c, m in batch]
        results = await run_batch(tasks)

        for combo_key, record in results:
            if record is not None:
                records.append(record)
                completed.add(combo_key)
                calls_made += 1
                since_checkpoint += 1
            else:
                errors += 1
                # Find the original args for retry
                for p, a, c, m in batch:
                    if (p["id"], a, m["name"]) == combo_key:
                        failed.append((p, a, c, m))
                        break

        # Checkpoint periodically
        if since_checkpoint >= CHECKPOINT_EVERY:
            save_checkpoint(records)
            since_checkpoint = 0
            done = len(completed)
            print(f"  Progress: {done}/{total_combos} "
                  f"({done/total_combos:.1%}) | "
                  f"+{calls_made} calls | "
                  f"{errors} errors")

    # Phase 3: Retry failed combos
    if failed:
        print("\n" + "-" * 80)
        print(f"PHASE 3: Retrying {len(failed)} failed combos")
        print("-" * 80)

        still_failed = 0
        for prompt, agg, comp, model in failed:
            combo_key = (prompt["id"], agg, model["name"])
            if combo_key in completed:
                continue

            sys_prompt = SYSTEM_PROMPTS.get(prompt["benchmark"], DEFAULT_SYSTEM_PROMPT)
            for attempt in range(MAX_RETRIES + 1):
                try:
                    llm_result = await call_llm_async(model, comp["compressed_text"], sys_prompt)
                    record = build_record(prompt, agg, comp, model, llm_result)
                    records.append(record)
                    completed.add(combo_key)
                    calls_made += 1
                    print(f"  RECOVERED {prompt['id']} / {model['name']} / agg={agg}")
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_DELAY * 2)
                    else:
                        print(f"  STILL FAILED {prompt['id']} / {model['name']} / agg={agg}: {e}")
                        still_failed += 1

        if still_failed:
            print(f"\n  {still_failed} combos could not be completed. "
                  f"Re-run the script to retry them.")

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
    print(f"  Missing: {total_combos - len(completed)}")
    print(f"  Saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
