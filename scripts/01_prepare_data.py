"""Load SQuAD 2.0 and FinQA (SEC 10-K) subsets for the grid search."""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset

from config import PROMPTS_PER_BENCHMARK

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_squad2(n: int) -> list[dict]:
    """Load n answerable questions from SQuAD 2.0 validation set."""
    ds = load_dataset("rajpurkar/squad_v2", split="validation")

    # Filter to answerable questions only (non-empty answers)
    answerable = [ex for ex in ds if len(ex["answers"]["text"]) > 0]

    random.seed(SEED)
    sampled = random.sample(answerable, min(n, len(answerable)))

    prompts = []
    for i, ex in enumerate(sampled):
        context = ex["context"]
        question = ex["question"]
        prompt_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        prompts.append({
            "id": f"squad2_{i:03d}",
            "benchmark": "squad2",
            "text": prompt_text,
            "ground_truth": ex["answers"]["text"][0],
            "context": context,
            "question": question,
        })

    return prompts



def load_finqa(n: int) -> list[dict]:
    """Load n questions from financial QA over SEC 10-K filings."""
    ds = load_dataset("virattt/financial-qa-10K", split="train")

    random.seed(SEED + 2)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    sampled = [ds[i] for i in indices]

    prompts = []
    for i, ex in enumerate(sampled):
        context = ex["context"]
        question = ex["question"]
        prompt_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        prompts.append({
            "id": f"finqa_{i:03d}",
            "benchmark": "finqa",
            "text": prompt_text,
            "ground_truth": ex["answer"],
            "context": context,
            "question": question,
        })

    return prompts


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Loading SQuAD 2.0 ({PROMPTS_PER_BENCHMARK} samples)...")
    squad = load_squad2(PROMPTS_PER_BENCHMARK)
    squad_path = os.path.join(DATA_DIR, "squad2_subset.json")
    with open(squad_path, "w") as f:
        json.dump(squad, f, indent=2)
    print(f"  Saved {len(squad)} prompts to {squad_path}")

    print(f"Loading FinQA / SEC 10-K ({PROMPTS_PER_BENCHMARK} samples)...")
    finqa = load_finqa(PROMPTS_PER_BENCHMARK)
    finqa_path = os.path.join(DATA_DIR, "finqa_subset.json")
    with open(finqa_path, "w") as f:
        json.dump(finqa, f, indent=2)
    print(f"  Saved {len(finqa)} prompts to {finqa_path}")

    total = len(squad) + len(finqa)
    print(f"\nTotal: {total} prompts ready for grid search.")


if __name__ == "__main__":
    main()
