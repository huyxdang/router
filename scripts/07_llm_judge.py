"""LLM-as-judge: use GPT-4o-mini to evaluate if each response is correct."""

from __future__ import annotations

import asyncio
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import OPENAI_API_KEY

import openai

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
GRID_PATH = os.path.join(RESULTS_DIR, "grid_results.parquet")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "grid_results_judged.parquet")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "judge_checkpoint.parquet")

JUDGE_MODEL = "gpt-4o-mini"
BATCH_SIZE = 20
CHECKPOINT_EVERY = 200
MAX_RPM = 450  # Stay under 500 RPM limit
MAX_RETRIES = 5

JUDGE_PROMPT = """You are an evaluation judge. Determine if the response correctly answers the question.

Ground truth answer: {ground_truth}

Model response: {response}

Is the model's response correct? It doesn't need to match exactly — it just needs to contain or convey the same answer as the ground truth.

Reply with ONLY "correct" or "incorrect"."""


async def judge_one(client: openai.AsyncOpenAI, semaphore: asyncio.Semaphore,
                    response_text: str, ground_truth: str) -> str | None:
    """Judge a single response with rate limiting and retries."""
    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            ground_truth=ground_truth,
                            response=response_text,
                        ),
                    }],
                    max_tokens=10,
                    temperature=0,
                )
                verdict = response.choices[0].message.content.strip().lower()
                if "correct" in verdict and "incorrect" not in verdict:
                    return "correct"
                return "incorrect"
            except openai.RateLimitError:
                wait = 2 ** attempt
                print(f"  Rate limited, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"  Judge error: {e}")
                return None
    print(f"  Failed after {MAX_RETRIES} retries")
    return None


async def main():
    print("=" * 80)
    print("LLM-AS-JUDGE EVALUATION")
    print("=" * 80)

    # Load grid results
    print("\n[1] Loading grid results...")
    df = pd.read_parquet(GRID_PATH)
    print(f"  {len(df)} records")

    # Load prompts to get ground truth
    import json
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    prompts = []
    for filename in ["squad2_subset.json", "finqa_subset.json"]:
        path = os.path.join(DATA_DIR, filename)
        with open(path) as f:
            prompts.extend(json.load(f))
    gt_map = {p["id"]: p["ground_truth"] for p in prompts}

    # Merge existing judgments from checkpoint (if any) into the full grid.
    # This handles dataset expansion: new rows get llm_judge=None automatically.
    df["llm_judge"] = None
    for ckpt in [CHECKPOINT_PATH, OUTPUT_PATH]:
        if os.path.exists(ckpt):
            existing = pd.read_parquet(ckpt)
            if "llm_judge" in existing.columns:
                key_cols = ["prompt_id", "model_name", "aggressiveness"]
                judged = existing[existing["llm_judge"].notna()][key_cols + ["llm_judge"]]
                if not judged.empty:
                    df = df.drop(columns=["llm_judge"]).merge(
                        judged, on=key_cols, how="left",
                    )
                    n_done = df["llm_judge"].notna().sum()
                    print(f"  Merged {n_done} existing judgments from {os.path.basename(ckpt)}")
                    break

    # Find rows that need judging
    needs_judging = df[df["llm_judge"].isna()].index.tolist()
    print(f"  Remaining: {len(needs_judging)}")

    if not needs_judging:
        print("  All records already judged!")
        df.to_parquet(OUTPUT_PATH, index=False)
        return

    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    # Semaphore limits concurrency to stay under RPM limit
    semaphore = asyncio.Semaphore(BATCH_SIZE)
    min_batch_time = BATCH_SIZE / MAX_RPM * 60  # seconds per batch to respect RPM

    # Process in batches
    print(f"\n[2] Judging {len(needs_judging)} records (batch_size={BATCH_SIZE}, max_rpm={MAX_RPM})...")
    done = 0
    errors = 0
    since_checkpoint = 0
    start = time.time()

    for batch_start in range(0, len(needs_judging), BATCH_SIZE):
        batch_start_time = time.time()
        batch_indices = needs_judging[batch_start:batch_start + BATCH_SIZE]

        tasks = []
        for idx in batch_indices:
            row = df.loc[idx]
            ground_truth = gt_map.get(row["prompt_id"], "")
            response_text = row.get("llm_response", "")
            if pd.isna(response_text):
                response_text = ""
            tasks.append(judge_one(client, semaphore, str(response_text), ground_truth))

        results = await asyncio.gather(*tasks)

        for idx, verdict in zip(batch_indices, results):
            if verdict is not None:
                df.at[idx, "llm_judge"] = verdict
                done += 1
                since_checkpoint += 1
            else:
                errors += 1

        # Throttle to stay under RPM limit
        elapsed_batch = time.time() - batch_start_time
        if elapsed_batch < min_batch_time:
            await asyncio.sleep(min_batch_time - elapsed_batch)

        if since_checkpoint >= CHECKPOINT_EVERY:
            df.to_parquet(CHECKPOINT_PATH, index=False)
            since_checkpoint = 0
            elapsed = time.time() - start
            total_done = len(needs_judging) - len(df[df["llm_judge"].isna()])
            print(f"  Progress: {total_done}/{len(needs_judging)} "
                  f"({total_done/len(needs_judging):.1%}) | "
                  f"{elapsed:.0f}s | {errors} errors")

    # Final save
    df.to_parquet(CHECKPOINT_PATH, index=False)
    df.to_parquet(OUTPUT_PATH, index=False)

    # Summary
    n_correct = (df["llm_judge"] == "correct").sum()
    n_incorrect = (df["llm_judge"] == "incorrect").sum()
    n_missing = df["llm_judge"].isna().sum()

    elapsed = time.time() - start

    print(f"\n" + "=" * 80)
    print("JUDGE COMPLETE")
    print("=" * 80)
    print(f"  Correct:   {n_correct} ({n_correct/len(df):.1%})")
    print(f"  Incorrect: {n_incorrect} ({n_incorrect/len(df):.1%})")
    print(f"  Missing:   {n_missing}")
    print(f"  Errors:    {errors}")
    print(f"  Time:      {elapsed:.0f}s")
    print(f"  Saved to:  {OUTPUT_PATH}")

    # Quick breakdown
    print(f"\n  By model:")
    for model in df["model_name"].unique():
        sub = df[df["model_name"] == model]
        acc = (sub["llm_judge"] == "correct").mean()
        print(f"    {model:<16s} accuracy={acc:.3f}")

    print(f"\n  By aggressiveness:")
    for agg in sorted(df["aggressiveness"].unique()):
        sub = df[df["aggressiveness"] == agg]
        acc = (sub["llm_judge"] == "correct").mean()
        print(f"    agg={agg:.1f}  accuracy={acc:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
