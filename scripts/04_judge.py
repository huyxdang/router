"""Judge grid search results via Modal GPU workers.

Usage:
  modal run scripts/04_judge.py
  modal run scripts/04_judge.py --batch-size 1000
  modal run scripts/04_judge.py --rejudge-all
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router.judge import app, Judge, parse_verdict, JUDGE_TEMPLATE


@app.local_entrypoint()
def main(batch_size: int = 500, rejudge_all: bool = False):
    import pandas as pd

    from config import RESULTS_DIR
    from router.data import load_ground_truths

    grid_path = os.path.join(str(RESULTS_DIR), "grid_results_judged.parquet")
    fallback_path = os.path.join(str(RESULTS_DIR), "grid_results.parquet")
    output_path = grid_path

    # Load data
    print("[1] Loading data...")
    if os.path.exists(grid_path) and not rejudge_all:
        df = pd.read_parquet(grid_path)
    elif os.path.exists(fallback_path):
        df = pd.read_parquet(fallback_path)
    else:
        print("  ERROR: No grid results found. Run 03_grid_search.py first.")
        return

    gt_map = load_ground_truths()

    if "llm_judge" not in df.columns or rejudge_all:
        df["llm_judge"] = None

    pending_mask = df["llm_judge"].isna()
    pending_idx = df.index[pending_mask].tolist()
    print(
        f"  Total: {len(df)} | "
        f"Judged: {(~pending_mask).sum()} | "
        f"Pending: {len(pending_idx)}"
    )

    if not pending_idx:
        print("  All rows already judged!")
        return

    # Build judge prompts
    print("[2] Building judge prompts...")
    prompts = []
    for idx in pending_idx:
        row = df.loc[idx]
        gt = gt_map.get(row["prompt_id"], "")
        response = str(row.get("llm_response", "") or "")
        prompts.append(
            JUDGE_TEMPLATE.format(ground_truth=gt, response=response)
        )

    batches = [
        prompts[i : i + batch_size]
        for i in range(0, len(prompts), batch_size)
    ]
    print(f"  {len(prompts)} prompts -> {len(batches)} batches of ~{batch_size}")

    # Judge on Modal
    print("[3] Judging on Modal...")
    judge = Judge()

    all_verdicts = []
    for i, batch_verdicts in enumerate(judge.judge_batch.map(batches)):
        parsed = [parse_verdict(v) for v in batch_verdicts]
        all_verdicts.extend(parsed)
        print(
            f"  Batch {i + 1}/{len(batches)} | "
            f"{len(all_verdicts)}/{len(prompts)} done"
        )

    # Merge results
    print("[4] Saving results...")
    for idx, verdict in zip(pending_idx, all_verdicts):
        df.at[idx, "llm_judge"] = verdict

    df.to_parquet(output_path, index=False)

    # Summary
    n_correct = (df["llm_judge"] == "correct").sum()
    n_incorrect = (df["llm_judge"] == "incorrect").sum()
    n_missing = df["llm_judge"].isna().sum()

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"  Correct:   {n_correct} ({n_correct / len(df):.1%})")
    print(f"  Incorrect: {n_incorrect} ({n_incorrect / len(df):.1%})")
    print(f"  Missing:   {n_missing}")
    print(f"  Saved to:  {output_path}")

    print("\n  By model:")
    for model in sorted(df["model_name"].unique()):
        sub = df[df["model_name"] == model]
        acc = (sub["llm_judge"] == "correct").mean()
        print(f"    {model:<16s} accuracy={acc:.3f}")

    print("\n  By aggressiveness:")
    for agg in sorted(df["aggressiveness"].unique()):
        sub = df[df["aggressiveness"] == agg]
        acc = (sub["llm_judge"] == "correct").mean()
        print(f"    agg={agg:.1f}  accuracy={acc:.3f}")
