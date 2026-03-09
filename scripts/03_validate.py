"""Quick validation: run a few prompts through the full pipeline to confirm everything works."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MODELS, AGGRESSIVENESS_LEVELS
from router.compress import compress
from router.llm import call_llm
from router.evaluate import exact_match, f1_score, contains_answer, compute_cost

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Use a small subset for validation
VALIDATION_PROMPTS = 2
VALIDATION_AGG_LEVELS = [0.1, 0.7]


def load_prompts(n: int) -> list[dict]:
    """Load first n prompts from each benchmark."""
    prompts = []
    for filename in ["squad2_subset.json", "finqa_subset.json"]:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found. Run 01_prepare_data.py first.")
            continue
        with open(path) as f:
            data = json.load(f)
        prompts.extend(data[:n])
    return prompts


def main():
    print("=" * 80)
    print("PIPELINE VALIDATION")
    print("=" * 80)

    # Step 1: Load prompts
    print("\n[1] Loading prompts...")
    prompts = load_prompts(VALIDATION_PROMPTS)
    if not prompts:
        print("No prompts found. Run scripts/01_prepare_data.py first.")
        return
    print(f"  Loaded {len(prompts)} prompts")

    # Step 2: Run pipeline
    print(f"\n[2] Running grid: {len(prompts)} prompts x {len(VALIDATION_AGG_LEVELS)} agg levels x {len(MODELS)} models")
    print(f"  Total calls: {len(prompts) * len(VALIDATION_AGG_LEVELS) * len(MODELS)}")
    print()

    results = []
    for prompt in prompts:
        for agg in VALIDATION_AGG_LEVELS:
            # Compress
            print(f"  Compressing {prompt['id']} @ agg={agg}...")
            comp = compress(prompt["text"], agg)
            print(f"    {comp['original_input_tokens']} → {comp['output_tokens']} tokens "
                  f"({comp['removal_rate']:.1%} removed)")

            for model in MODELS:
                # Call LLM
                print(f"    Calling {model['name']}...")
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

                results.append({
                    "prompt_id": prompt["id"],
                    "model": model["name"],
                    "agg": agg,
                    "exact_match": em,
                    "f1": f1,
                    "contains_answer": ca,
                    "total_cost": cost["total_cost_usd"],
                    "latency": llm_result["latency"],
                    "response_preview": llm_result["response_text"][:80],
                })

    # Step 3: Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"{'Prompt':<12} {'Model':<16} {'Agg':>4} {'EM':>3} {'F1':>5} {'CA':>3} {'Cost':>8} {'Latency':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r['prompt_id']:<12} {r['model']:<16} {r['agg']:>4.1f} {r['exact_match']:>3} "
              f"{r['f1']:>5.2f} {r['contains_answer']:>3} ${r['total_cost']:>7.5f} {r['latency']:>6.2f}s")

    print("\n" + "-" * 80)
    print("Response previews:")
    for r in results:
        print(f"  [{r['prompt_id']} / {r['model']} / agg={r['agg']}]: {r['response_preview']}...")

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
