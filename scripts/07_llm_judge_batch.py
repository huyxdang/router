"""LLM-as-judge using OpenAI Batch API (50% cheaper, no rate limits).

Usage:
  python scripts/07_llm_judge_batch.py submit    # Create and submit batch
  python scripts/07_llm_judge_batch.py status     # Check batch status
  python scripts/07_llm_judge_batch.py download   # Download results and merge
"""

from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import OPENAI_API_KEY

import openai

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
GRID_PATH = os.path.join(RESULTS_DIR, "grid_results.parquet")
BATCH_INPUT_PATH = os.path.join(RESULTS_DIR, "judge_batch_input.jsonl")
BATCH_META_PATH = os.path.join(RESULTS_DIR, "judge_batch_meta.json")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "grid_results_judged.parquet")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

JUDGE_MODEL = "gpt-4o-mini"

JUDGE_PROMPT = """You are an evaluation judge. Determine if the response correctly answers the question.

Ground truth answer: {ground_truth}

Model response: {response}

Is the model's response correct? It doesn't need to match exactly — it just needs to contain or convey the same answer as the ground truth.

Reply with ONLY "correct" or "incorrect"."""


def load_ground_truths() -> dict:
    gt_map = {}
    for filename in ["squad2_subset.json", "finqa_subset.json"]:
        path = os.path.join(DATA_DIR, filename)
        with open(path) as f:
            for p in json.load(f):
                gt_map[p["id"]] = p["ground_truth"]
    return gt_map


def _merge_existing_judgments(df: pd.DataFrame) -> pd.DataFrame:
    """Merge existing judgments from checkpoint/output into df."""
    df["llm_judge"] = None
    key_cols = ["prompt_id", "model_name", "aggressiveness"]
    for path in [OUTPUT_PATH, os.path.join(RESULTS_DIR, "judge_checkpoint.parquet")]:
        if os.path.exists(path):
            existing = pd.read_parquet(path)
            if "llm_judge" in existing.columns:
                judged = existing[existing["llm_judge"].notna()][key_cols + ["llm_judge"]]
                if not judged.empty:
                    df = df.drop(columns=["llm_judge"]).merge(
                        judged, on=key_cols, how="left",
                    )
                    n = df["llm_judge"].notna().sum()
                    print(f"  Merged {n} existing judgments from {os.path.basename(path)}")
                    break
    return df


def submit():
    """Create JSONL input file and submit batch to OpenAI (only unjudged rows)."""
    print("[1] Loading grid results...")
    df = pd.read_parquet(GRID_PATH)
    gt_map = load_ground_truths()
    print(f"  {len(df)} total records")

    # Merge existing judgments so we only submit new rows
    df = _merge_existing_judgments(df)
    to_judge = df[df["llm_judge"].isna()]
    print(f"  {len(to_judge)} records need judging")

    if to_judge.empty:
        print("  All records already judged!")
        return

    # Build JSONL (only unjudged rows)
    print("[2] Building batch input file...")
    with open(BATCH_INPUT_PATH, "w") as f:
        for idx, row in to_judge.iterrows():
            ground_truth = gt_map.get(row["prompt_id"], "")
            response_text = str(row.get("llm_response", "") or "")

            request = {
                "custom_id": f"row-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": JUDGE_MODEL,
                    "messages": [{
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            ground_truth=ground_truth,
                            response=response_text,
                        ),
                    }],
                    "max_tokens": 10,
                    "temperature": 0,
                },
            }
            f.write(json.dumps(request) + "\n")

    print(f"  Written {len(to_judge)} requests to {BATCH_INPUT_PATH}")

    # Upload file
    print("[3] Uploading to OpenAI...")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    with open(BATCH_INPUT_PATH, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    print(f"  File ID: {file_obj.id}")

    # Create batch
    print("[4] Creating batch...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "LLM judge for grid search results"},
    )

    print(f"  Batch ID: {batch.id}")
    print(f"  Status:   {batch.status}")

    # Save metadata
    meta = {
        "batch_id": batch.id,
        "file_id": file_obj.id,
        "n_records": len(to_judge),
        "n_total": len(df),
        "submitted_at": time.time(),
    }
    with open(BATCH_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Batch submitted! Run 'python scripts/07_llm_judge_batch.py status' to check progress.")


def status():
    """Check batch status."""
    with open(BATCH_META_PATH) as f:
        meta = json.load(f)

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    batch = client.batches.retrieve(meta["batch_id"])

    elapsed = time.time() - meta["submitted_at"]
    print(f"Batch ID:    {batch.id}")
    print(f"Status:      {batch.status}")
    print(f"Elapsed:     {elapsed:.0f}s ({elapsed/60:.1f}m)")

    if batch.request_counts:
        print(f"Total:       {batch.request_counts.total}")
        print(f"Completed:   {batch.request_counts.completed}")
        print(f"Failed:      {batch.request_counts.failed}")

    if batch.status == "completed":
        print(f"\nBatch complete! Run 'python scripts/07_llm_judge_batch.py download' to get results.")
    elif batch.status == "failed":
        print(f"\nBatch failed!")
        if batch.errors:
            for error in batch.errors.data:
                print(f"  Error: {error.message}")


def download():
    """Download batch results and merge with grid data."""
    with open(BATCH_META_PATH) as f:
        meta = json.load(f)

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    batch = client.batches.retrieve(meta["batch_id"])

    if batch.status != "completed":
        print(f"Batch not complete yet. Status: {batch.status}")
        return

    # Download output file
    print("[1] Downloading results...")
    output_file_id = batch.output_file_id
    content = client.files.content(output_file_id)
    results_text = content.text

    # Parse results
    print("[2] Parsing judge verdicts...")
    verdicts = {}
    for line in results_text.strip().split("\n"):
        result = json.loads(line)
        custom_id = result["custom_id"]
        idx = int(custom_id.split("-")[1])

        if result["response"]["status_code"] == 200:
            body = result["response"]["body"]
            text = body["choices"][0]["message"]["content"].strip().lower()
            if "correct" in text and "incorrect" not in text:
                verdicts[idx] = "correct"
            else:
                verdicts[idx] = "incorrect"
        else:
            verdicts[idx] = None

    print(f"  Parsed {len(verdicts)} verdicts")

    # Merge with grid results (preserve existing judgments + add new ones)
    print("[3] Merging with grid results...")
    df = pd.read_parquet(GRID_PATH)
    df = _merge_existing_judgments(df)

    # Apply new verdicts from this batch
    new_count = 0
    for idx, verdict in verdicts.items():
        if verdict is not None and idx < len(df):
            df.at[idx, "llm_judge"] = verdict
            new_count += 1
    print(f"  Applied {new_count} new verdicts from batch")

    n_correct = (df["llm_judge"] == "correct").sum()
    n_incorrect = (df["llm_judge"] == "incorrect").sum()
    n_missing = df["llm_judge"].isna().sum()

    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n" + "=" * 80)
    print("JUDGE RESULTS")
    print("=" * 80)
    print(f"  Correct:   {n_correct} ({n_correct/len(df):.1%})")
    print(f"  Incorrect: {n_incorrect} ({n_incorrect/len(df):.1%})")
    print(f"  Missing:   {n_missing}")
    print(f"  Saved to:  {OUTPUT_PATH}")

    # Breakdown
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
    if len(sys.argv) < 2:
        print("Usage: python scripts/07_llm_judge_batch.py [submit|status|download]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "submit":
        submit()
    elif cmd == "status":
        status()
    elif cmd == "download":
        download()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
