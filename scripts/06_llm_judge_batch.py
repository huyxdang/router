"""LLM-as-judge using OpenAI Batch API (50% cheaper, no rate limits).

Usage:
  python scripts/03_llm_judge_batch.py submit [max_records]  # Create and submit batch (optional chunk size)
  python scripts/03_llm_judge_batch.py status    # Check batch status
  python scripts/03_llm_judge_batch.py download  # Download results and merge
  python scripts/03_llm_judge_batch.py run [max_records] [poll_seconds]  # Auto-loop submit -> wait -> download
  python scripts/03_llm_judge_batch.py direct [max_records] [concurrency]  # Direct async judging (fast path)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import OPENAI_API_KEY, RESULTS_DIR, JUDGE_MODEL
from router.data import load_ground_truths
from router.evaluate import JUDGE_PROMPT, parse_judge_verdict

import openai

GRID_PATH = os.path.join(str(RESULTS_DIR), "grid_results.parquet")
BATCH_INPUT_PATH = os.path.join(str(RESULTS_DIR), "judge_batch_input.jsonl")
BATCH_META_PATH = os.path.join(str(RESULTS_DIR), "judge_batch_meta.json")
OUTPUT_PATH = os.path.join(str(RESULTS_DIR), "grid_results_judged.parquet")
DEFAULT_MAX_RECORDS = int(os.environ.get("JUDGE_BATCH_MAX_RECORDS", "1200"))
DEFAULT_POLL_SECONDS = int(os.environ.get("JUDGE_BATCH_POLL_SECONDS", "30"))
DEFAULT_MAX_RUN_CYCLES = int(os.environ.get("JUDGE_BATCH_MAX_RUN_CYCLES", "500"))
DIRECT_CHECKPOINT_PATH = os.path.join(str(RESULTS_DIR), "judge_checkpoint.parquet")
DEFAULT_DIRECT_CONCURRENCY = int(os.environ.get("JUDGE_DIRECT_CONCURRENCY", "30"))
DEFAULT_DIRECT_SAVE_EVERY = int(os.environ.get("JUDGE_DIRECT_SAVE_EVERY", "100"))
DEFAULT_DIRECT_MAX_RETRIES = int(os.environ.get("JUDGE_DIRECT_MAX_RETRIES", "5"))
DEFAULT_DIRECT_RETRY_DELAY = float(os.environ.get("JUDGE_DIRECT_RETRY_DELAY", "2.0"))


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


def _load_meta() -> dict:
    if not os.path.exists(BATCH_META_PATH):
        raise FileNotFoundError(
            f"{BATCH_META_PATH} not found. Run submit first."
        )
    with open(BATCH_META_PATH) as f:
        return json.load(f)


def _get_batch():
    meta = _load_meta()
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    batch = client.batches.retrieve(meta["batch_id"])
    return client, meta, batch


def _batch_error_messages(batch) -> list[str]:
    errors = []
    if getattr(batch, "errors", None) and getattr(batch.errors, "data", None):
        for err in batch.errors.data:
            msg = getattr(err, "message", None)
            if msg:
                errors.append(msg)
    return errors


def _is_enqueued_limit_error_text(text: str) -> bool:
    t = text.lower()
    return "enqueued token limit" in t or "enqueued_tokens" in t


def _is_enqueued_limit_error_batch(batch) -> bool:
    for msg in _batch_error_messages(batch):
        if _is_enqueued_limit_error_text(msg):
            return True
    return False


def _print_judge_summary(df: pd.DataFrame):
    n_correct = (df["llm_judge"] == "correct").sum()
    n_incorrect = (df["llm_judge"] == "incorrect").sum()
    n_missing = df["llm_judge"].isna().sum()

    print("\n" + "=" * 80)
    print("JUDGE RESULTS")
    print("=" * 80)
    print(f"  Correct:   {n_correct} ({n_correct/len(df):.1%})")
    print(f"  Incorrect: {n_incorrect} ({n_incorrect/len(df):.1%})")
    print(f"  Missing:   {n_missing}")
    print(f"  Saved to:  {OUTPUT_PATH}")

    print("\n  By model:")
    for model in df["model_name"].unique():
        sub = df[df["model_name"] == model]
        acc = (sub["llm_judge"] == "correct").mean()
        print(f"    {model:<16s} accuracy={acc:.3f}")

    print("\n  By aggressiveness:")
    for agg in sorted(df["aggressiveness"].unique()):
        sub = df[df["aggressiveness"] == agg]
        acc = (sub["llm_judge"] == "correct").mean()
        print(f"    agg={agg:.1f}  accuracy={acc:.3f}")


def submit(max_records: int | None = None):
    """Create JSONL input file and submit batch to OpenAI (only unjudged rows)."""
    if max_records is None:
        max_records = DEFAULT_MAX_RECORDS

    print("[1] Loading grid results...")
    df = pd.read_parquet(GRID_PATH)
    gt_map = load_ground_truths()
    print(f"  {len(df)} total records")

    # Merge existing judgments so we only submit new rows
    df = _merge_existing_judgments(df)
    to_judge = df[df["llm_judge"].isna()]
    n_pending = len(to_judge)
    print(f"  {n_pending} records need judging")

    if max_records and n_pending > max_records:
        to_judge = to_judge.head(max_records)
        print(f"  Limiting submission to {len(to_judge)} records "
              f"(set via max_records={max_records})")

    if to_judge.empty:
        print("  All records already judged!")
        return {
            "all_judged": True,
            "submitted": 0,
            "batch_id": None,
            "n_pending": 0,
        }

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
        "n_remaining_estimate": max(n_pending - len(to_judge), 0),
        "submitted_at": time.time(),
    }
    with open(BATCH_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Batch submitted! Run 'python scripts/03_llm_judge_batch.py status' to check progress.")
    return {
        "all_judged": False,
        "submitted": len(to_judge),
        "batch_id": batch.id,
        "n_pending": n_pending,
    }


def status():
    """Check batch status."""
    _, meta, batch = _get_batch()

    elapsed = time.time() - meta["submitted_at"]
    print(f"Batch ID:    {batch.id}")
    print(f"Status:      {batch.status}")
    print(f"Elapsed:     {elapsed:.0f}s ({elapsed/60:.1f}m)")

    if batch.request_counts:
        print(f"Total:       {batch.request_counts.total}")
        print(f"Completed:   {batch.request_counts.completed}")
        print(f"Failed:      {batch.request_counts.failed}")

    if batch.status == "completed":
        print(f"\nBatch complete! Run 'python scripts/03_llm_judge_batch.py download' to get results.")
    elif batch.status == "failed":
        print(f"\nBatch failed!")
        for message in _batch_error_messages(batch):
            print(f"  Error: {message}")
    return batch.status


def download():
    """Download batch results and merge with grid data."""
    client, _, batch = _get_batch()

    if batch.status != "completed":
        print(f"Batch not complete yet. Status: {batch.status}")
        return False

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
            text = body["choices"][0]["message"]["content"]
            verdicts[idx] = parse_judge_verdict(text)
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

    df.to_parquet(OUTPUT_PATH, index=False)
    _print_judge_summary(df)
    return True


def run(max_records: int | None = None, poll_seconds: int = DEFAULT_POLL_SECONDS):
    """Automatically process all remaining rows in chunked batches."""
    if max_records is None:
        max_records = DEFAULT_MAX_RECORDS
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be a positive integer")

    print("=" * 80)
    print("AUTO JUDGE RUN")
    print("=" * 80)
    print(f"  chunk_size={max_records}  poll_seconds={poll_seconds}")

    for cycle in range(1, DEFAULT_MAX_RUN_CYCLES + 1):
        print(f"\n[Cycle {cycle}]")
        try:
            submit_info = submit(max_records)
        except Exception as e:
            msg = str(e)
            if _is_enqueued_limit_error_text(msg):
                if max_records > 1:
                    max_records = max(1, max_records // 2)
                    print(f"  Token queue limit hit. Reducing chunk_size to {max_records} and retrying...")
                    time.sleep(poll_seconds)
                    continue
            raise

        if submit_info["all_judged"]:
            print("\nAll rows are judged. Nothing left to submit.")
            return

        while True:
            time.sleep(poll_seconds)
            _, _, batch = _get_batch()
            counts = getattr(batch, "request_counts", None)
            if counts:
                print(f"  Batch {batch.id}: {batch.status} "
                      f"(completed={counts.completed}, failed={counts.failed}, total={counts.total})")
            else:
                print(f"  Batch {batch.id}: {batch.status}")

            if batch.status == "completed":
                ok = download()
                if not ok:
                    raise RuntimeError("Download failed after completed status")
                break

            if batch.status == "failed":
                messages = _batch_error_messages(batch)
                for message in messages:
                    print(f"  Error: {message}")
                if _is_enqueued_limit_error_batch(batch) and max_records > 1:
                    max_records = max(1, max_records // 2)
                    print(f"  Token queue limit hit. Reducing chunk_size to {max_records} for next cycle.")
                    break
                raise RuntimeError(f"Batch failed: {messages or 'unknown error'}")

            if batch.status in {"expired", "cancelled"}:
                raise RuntimeError(f"Batch ended with status={batch.status}")

    raise RuntimeError(
        f"Reached max cycles ({DEFAULT_MAX_RUN_CYCLES}) before finishing."
    )


async def _direct_async(max_records: int | None = None,
                        concurrency: int = DEFAULT_DIRECT_CONCURRENCY):
    if concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")

    print("=" * 80)
    print("DIRECT JUDGE RUN")
    print("=" * 80)
    print(f"  concurrency={concurrency}")

    print("[1] Loading grid results...")
    df = pd.read_parquet(GRID_PATH)
    gt_map = load_ground_truths()
    print(f"  {len(df)} total records")

    df = _merge_existing_judgments(df)
    pending_idx = df.index[df["llm_judge"].isna()].tolist()
    if max_records is not None:
        pending_idx = pending_idx[:max_records]

    if not pending_idx:
        print("  All records already judged!")
        if os.path.exists(OUTPUT_PATH):
            _print_judge_summary(pd.read_parquet(OUTPUT_PATH))
        return

    print(f"  Pending in this run: {len(pending_idx)}")

    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(concurrency)
    processed = 0
    success = 0
    failed = 0

    async def _judge_one(idx: int):
        row = df.loc[idx]
        ground_truth = gt_map.get(row["prompt_id"], "")
        response_text = str(row.get("llm_response", "") or "")
        prompt = JUDGE_PROMPT.format(ground_truth=ground_truth, response=response_text)

        async with semaphore:
            for attempt in range(DEFAULT_DIRECT_MAX_RETRIES + 1):
                try:
                    resp = await client.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=10,
                        temperature=0,
                    )
                    verdict = parse_judge_verdict(resp.choices[0].message.content)
                    return idx, verdict
                except Exception as e:
                    if attempt < DEFAULT_DIRECT_MAX_RETRIES:
                        delay = DEFAULT_DIRECT_RETRY_DELAY * (2 ** attempt)
                        if "429" in str(e) or "rate" in str(e).lower():
                            delay = max(delay, 5.0)
                        await asyncio.sleep(delay)
                    else:
                        return idx, None

    tasks = [asyncio.create_task(_judge_one(idx)) for idx in pending_idx]
    total = len(tasks)

    for fut in asyncio.as_completed(tasks):
        idx, verdict = await fut
        processed += 1

        if verdict is None:
            failed += 1
        else:
            df.at[idx, "llm_judge"] = verdict
            success += 1

        if processed % DEFAULT_DIRECT_SAVE_EVERY == 0 or processed == total:
            df.to_parquet(DIRECT_CHECKPOINT_PATH, index=False)
            df.to_parquet(OUTPUT_PATH, index=False)
            print(
                f"  Progress: {processed}/{total} | success={success} failed={failed}",
                flush=True,
            )

    print(f"\n[2] Direct run complete. success={success}, failed={failed}")
    _print_judge_summary(pd.read_parquet(OUTPUT_PATH))


def direct(max_records: int | None = None,
           concurrency: int = DEFAULT_DIRECT_CONCURRENCY):
    asyncio.run(_direct_async(max_records=max_records, concurrency=concurrency))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_llm_judge_batch.py "
              "[submit [max_records]|status|download|run [max_records] [poll_seconds]|"
              "direct [max_records] [concurrency]]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "submit":
        max_records = None
        if len(sys.argv) >= 3:
            try:
                max_records = int(sys.argv[2])
                if max_records <= 0:
                    raise ValueError
            except ValueError:
                print("submit max_records must be a positive integer")
                sys.exit(1)
        submit(max_records)
    elif cmd == "status":
        status()
    elif cmd == "download":
        download()
    elif cmd == "run":
        max_records = None
        poll_seconds = DEFAULT_POLL_SECONDS
        if len(sys.argv) >= 3:
            try:
                max_records = int(sys.argv[2])
                if max_records <= 0:
                    raise ValueError
            except ValueError:
                print("run max_records must be a positive integer")
                sys.exit(1)
        if len(sys.argv) >= 4:
            try:
                poll_seconds = int(sys.argv[3])
                if poll_seconds <= 0:
                    raise ValueError
            except ValueError:
                print("run poll_seconds must be a positive integer")
                sys.exit(1)
        run(max_records=max_records, poll_seconds=poll_seconds)
    elif cmd == "direct":
        max_records = None
        concurrency = DEFAULT_DIRECT_CONCURRENCY
        if len(sys.argv) >= 3:
            try:
                max_records = int(sys.argv[2])
                if max_records <= 0:
                    raise ValueError
            except ValueError:
                print("direct max_records must be a positive integer")
                sys.exit(1)
        if len(sys.argv) >= 4:
            try:
                concurrency = int(sys.argv[3])
                if concurrency <= 0:
                    raise ValueError
            except ValueError:
                print("direct concurrency must be a positive integer")
                sys.exit(1)
        direct(max_records=max_records, concurrency=concurrency)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
