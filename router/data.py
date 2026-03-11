"""Shared data loading utilities."""

from __future__ import annotations

import json
import os

from config import DATA_DIR, BENCHMARKS


def load_prompts(benchmarks: list[str] | None = None) -> list[dict]:
    """Load prompts from benchmark JSON files.

    Args:
        benchmarks: List of benchmark names to load. Defaults to BENCHMARKS from config.
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS

    prompts = []
    for bench in benchmarks:
        path = os.path.join(str(DATA_DIR), f"{bench}_subset.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            prompts.extend(json.load(f))
    return prompts


def load_ground_truths(benchmarks: list[str] | None = None) -> dict[str, str]:
    """Load ground truth answers keyed by prompt ID."""
    prompts = load_prompts(benchmarks)
    return {p["id"]: p["ground_truth"] for p in prompts}
