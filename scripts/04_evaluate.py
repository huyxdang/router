"""Evaluate the router against baselines: deferral curves, AUC, QNC."""

from __future__ import annotations

import asyncio
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MODELS, RESULTS_DIR, BENCHMARKS, DATA_DIR, OPENAI_API_KEY, JUDGE_MODEL,
    EVAL_BENCHMARKS, SYSTEM_PROMPTS, DEFAULT_SYSTEM_PROMPT,
    AGGRESSIVENESS_LEVELS, BATCH_SIZE, get_model_by_name, OPENROUTER_MODEL_IDS,
)
from router.data import load_prompts, load_ground_truths
from router.clustering import compute_cluster_stats_minimal
from router.scoring import (
    evaluate_router, compute_deferral_curve, compute_auc, compute_qnc,
    score_candidates,
)
from router.evaluate import JUDGE_PROMPT, parse_judge_verdict, compute_cost
from router.embeddings import embed_and_cache
from router.compress import compress, compress_async
from router.llm import call_llm, call_llm_async

# Semaphores — initialized in main() to bind to the correct event loop
_api_semaphore = None
_judge_semaphore = None

SPLITS_PATH = os.path.join(str(RESULTS_DIR), "router_splits.json")
GRID_PATH = os.path.join(str(RESULTS_DIR), "grid_results_clustered.parquet")


def evaluate_fixed_strategy(df_test: pd.DataFrame, model_name: str,
                            aggressiveness: float) -> dict:
    """Evaluate a fixed (model, agg) strategy on test data."""
    sub = df_test[
        (df_test["model_name"] == model_name) &
        (df_test["aggressiveness"] == aggressiveness)
    ]
    if len(sub) == 0:
        return {"accuracy": 0.0, "cost": 0.0, "count": 0}

    return {
        "accuracy": sub["llm_judge_correct"].mean(),
        "cost": sub["total_cost_usd"].mean(),
        "count": len(sub),
    }


async def evaluate_openrouter_baseline(df_test: pd.DataFrame) -> dict:
    """Evaluate OpenRouter auto-router on test prompts (no compression).

    Sends each test prompt to OpenRouter restricted to our model pool.
    Returns accuracy and cost. Requires live API calls.
    """
    import openai as oai
    from router.llm import call_openrouter_async

    allowed_model_ids = [m["id"] for m in MODELS]
    or_to_model_config = {
        OPENROUTER_MODEL_IDS[m["id"]]: m
        for m in MODELS
        if m["id"] in OPENROUTER_MODEL_IDS
    }

    prompt_lookup = {p["id"]: p["text"] for p in load_prompts()}
    ground_truths = load_ground_truths()

    test_prompt_ids = df_test["prompt_id"].unique()

    # Phase 1: Call OpenRouter for all test prompts (semaphore-throttled)
    print("    Calling OpenRouter API for test prompts...")

    async def _call_one(pid):
        text = prompt_lookup.get(pid)
        if text is None:
            return None
        async with _api_semaphore:
            try:
                result = await call_openrouter_async(text, allowed_model_ids)
                return {
                    "prompt_id": pid,
                    "response_text": result["response_text"],
                    "model_used": result["model_used"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "latency": result["latency"],
                }
            except Exception as e:
                print(f"    OpenRouter error for {pid}: {e}")
                return None

    tasks = [_call_one(pid) for pid in test_prompt_ids]
    raw_results = await asyncio.gather(*tasks)
    or_results = [r for r in raw_results if r is not None]

    if not or_results:
        return {"accuracy": 0.0, "cost": 0.0, "count": 0}

    # Phase 2: Judge OpenRouter responses (semaphore-throttled)
    judge_client = oai.AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def _judge_one(r):
        gt = ground_truths.get(r["prompt_id"], "")
        async with _judge_semaphore:
            try:
                judge_resp = await judge_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                        ground_truth=gt, response=r["response_text"]
                    )}],
                    max_tokens=10,
                    temperature=0,
                )
                verdict = parse_judge_verdict(judge_resp.choices[0].message.content)
                return verdict == "correct"
            except Exception as e:
                print(f"    Judge error: {e}")
                return False

    judge_tasks = [_judge_one(r) for r in or_results]
    verdicts = await asyncio.gather(*judge_tasks)

    correct = sum(verdicts)
    total = len(or_results)
    total_cost = 0.0
    unknown_model_count = 0
    for r in or_results:
        model_used = (r.get("model_used") or "").split(":")[0]
        model_config = or_to_model_config.get(model_used)
        if model_config is None:
            unknown_model_count += 1
            continue
        cost = compute_cost(model_config, r["input_tokens"], r["output_tokens"], tokens_removed=0)
        total_cost += cost["total_cost_usd"]

    cost_count = total - unknown_model_count
    mean_cost = total_cost / cost_count if cost_count else 0.0
    if unknown_model_count:
        print(f"    WARNING: {unknown_model_count}/{total} OpenRouter responses had unknown model IDs; cost excludes them.")

    return {
        "accuracy": correct / total if total else 0.0,
        "cost": mean_cost,
        "count": total,
    }


async def evaluate_financebench(cluster_stats, kmeans, lambda_values=[0, 1, 10, 100]):
    """Run the router live on FinanceBench (out-of-domain eval).

    Since FinanceBench isn't in the grid search, we do live inference:
    embed → cluster → pick (model, agg) → compress → call LLM → judge.
    All prompts processed concurrently with semaphore throttling.
    """
    import openai as oai

    # Load FinanceBench prompts
    fb_prompts = []
    for bench in EVAL_BENCHMARKS:
        path = os.path.join(str(DATA_DIR), f"{bench}_subset.json")
        if not os.path.exists(path):
            print(f"    {path} not found, skipping")
            continue
        with open(path) as f:
            fb_prompts.extend(json.load(f))

    if not fb_prompts:
        return {}

    print(f"    Loaded {len(fb_prompts)} FinanceBench prompts")

    # Embed and assign clusters
    fb_ids = [p["id"] for p in fb_prompts]
    fb_texts = [p["text"] for p in fb_prompts]
    fb_embeddings = embed_and_cache(fb_ids, fb_texts)
    fb_clusters = kmeans.predict(fb_embeddings)

    judge_client = oai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    results_by_lambda = {}

    for lam in lambda_values:
        async def _process_prompt(i, prompt, _lam=lam):
            cluster_id = int(fb_clusters[i])

            # Get candidates for this cluster
            candidates = cluster_stats[
                cluster_stats["cluster_id"] == cluster_id
            ].copy()

            if len(candidates) == 0:
                return None

            best = score_candidates(candidates, _lam)
            chosen_model_name = best["model_name"]
            chosen_agg = float(best["aggressiveness"])

            # Compress + LLM (semaphore-throttled)
            async with _api_semaphore:
                try:
                    comp = await compress_async(prompt["text"], chosen_agg)
                except Exception as e:
                    print(f"    Compress error {prompt['id']}: {e}")
                    return None

                model_config = get_model_by_name(chosen_model_name)
                sys_prompt = SYSTEM_PROMPTS.get(prompt["benchmark"], DEFAULT_SYSTEM_PROMPT)
                try:
                    llm_result = await call_llm_async(model_config, comp["compressed_text"], sys_prompt)
                except Exception as e:
                    print(f"    LLM error {prompt['id']}: {e}")
                    return None

            # Judge (separate semaphore)
            async with _judge_semaphore:
                try:
                    judge_resp = await judge_client.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                            ground_truth=prompt["ground_truth"],
                            response=llm_result["response_text"],
                        )}],
                        max_tokens=10,
                        temperature=0,
                    )
                    verdict = parse_judge_verdict(judge_resp.choices[0].message.content)
                    is_correct = verdict == "correct"
                except Exception as e:
                    print(f"    Judge error: {e}")
                    is_correct = False

            cost_info = compute_cost(
                model_config,
                llm_result["input_tokens"],
                llm_result["output_tokens"],
                comp["tokens_removed"],
            )
            return {"correct": is_correct, "cost": cost_info["total_cost_usd"]}

        tasks = [_process_prompt(i, p) for i, p in enumerate(fb_prompts)]
        results = await asyncio.gather(*tasks)

        correct = 0
        total = 0
        total_cost = 0.0
        for r in results:
            if r is not None:
                if r["correct"]:
                    correct += 1
                total_cost += r["cost"]
                total += 1

        results_by_lambda[lam] = {
            "accuracy": correct / total if total else 0.0,
            "cost": total_cost / total if total else 0.0,
            "count": total,
        }
        print(f"    lambda={lam}: acc={results_by_lambda[lam]['accuracy']:.3f}  "
              f"cost=${results_by_lambda[lam]['cost']:.6f}  n={total}")

    return results_by_lambda


async def main():
    global _api_semaphore, _judge_semaphore
    _api_semaphore = asyncio.Semaphore(BATCH_SIZE)
    _judge_semaphore = asyncio.Semaphore(15)

    print("=" * 80)
    print("ROUTER EVALUATION")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_parquet(GRID_PATH)
    print(f"  {len(df)} records, {df['prompt_id'].nunique()} prompts")

    # Load train/test split
    print("\n[2] Loading train/test split...")
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    test_ids = set(splits["test_ids"])
    train_val_ids = set(splits["train_ids"] + splits["val_ids"])

    df_train = df[df["prompt_id"].isin(train_val_ids)]
    df_test = df[df["prompt_id"].isin(test_ids)]
    print(f"  Train+Val: {df_train['prompt_id'].nunique()} prompts ({len(df_train)} records)")
    print(f"  Test:      {df_test['prompt_id'].nunique()} prompts ({len(df_test)} records)")

    # Build cluster stats from TRAIN only
    print("\n[3] Building cluster stats from train set...")
    cluster_stats = compute_cluster_stats_minimal(df_train)
    print(f"  {len(cluster_stats)} (cluster, model, agg) combos")

    # ===== BASELINE 1: GPT-5.4 Only (no compression) =====
    print("\n[4] Evaluating baselines...")

    print("\n  --- Baseline 1: GPT-5.4 Only (no router, no compression) ---")
    gpt54_baseline = evaluate_fixed_strategy(df_test, "gpt-5.4", 0.0)
    print(f"  GPT-5.4 acc={gpt54_baseline['accuracy']:.3f}  cost=${gpt54_baseline['cost']:.6f}")

    # ===== BASELINE 2: OpenRouter (same 6-model pool, no compression) =====
    print("\n  --- Baseline 2: OpenRouter (same pool, no compression) ---")
    openrouter_baseline = await evaluate_openrouter_baseline(df_test)
    print(f"  OpenRouter acc={openrouter_baseline['accuracy']:.3f}  "
          f"cost=${openrouter_baseline['cost']:.6f}  "
          f"n={openrouter_baseline['count']}")

    # ===== BASELINE 3: UniRoute No Compression (our router, agg=0.0 only) =====
    print("\n  --- Baseline 3: UniRoute No Compression (agg=0.0 only) ---")
    no_compress_curve = compute_deferral_curve(
        df_test, cluster_stats, models=None, agg_filter=[0.0]
    )
    auc_no_compress = compute_auc(no_compress_curve)
    nc_quality = evaluate_router(df_test, cluster_stats, 0.0, agg_filter=[0.0])
    print(f"  No-compress (lambda=0) acc={nc_quality['accuracy']:.3f}  "
          f"cost=${nc_quality['cost']:.6f}")
    print(f"  No-compress AUC={auc_no_compress:.6f}")

    # ===== OUR ROUTER (full: all models x all compression tiers) =====
    print("\n[5] Computing router deferral curve (full)...")
    router_curve = compute_deferral_curve(df_test, cluster_stats)
    auc_router = compute_auc(router_curve)

    print(f"\n[6] Results summary...")
    print(f"  Router AUC (full):          {auc_router:.6f}")
    print(f"  Router AUC (no compress):   {auc_no_compress:.6f}")
    if auc_no_compress > 0:
        auc_gain = (auc_router - auc_no_compress) / auc_no_compress * 100
        print(f"  Compression AUC gain:       {auc_gain:+.1f}%")

    # QNC
    qnc = compute_qnc(router_curve, gpt54_baseline["accuracy"])
    print(f"\n  GPT-5.4 baseline acc:       {gpt54_baseline['accuracy']:.3f}")
    print(f"  GPT-5.4 baseline cost:      ${gpt54_baseline['cost']:.6f}")
    if qnc is not None:
        print(f"  QNC (router cost to match): ${qnc:.6f}")
        if gpt54_baseline["cost"] > 0:
            savings = (1 - qnc / gpt54_baseline["cost"]) * 100
            print(f"  Cost savings vs GPT-5.4:    {savings:.1f}%")
    else:
        print(f"  QNC: router cannot match GPT-5.4 accuracy")

    # Router curve summary
    print(f"\n  Router deferral curve (sampled):")
    print(f"  {'lambda':>8s} {'Acc':>6s} {'Cost':>10s}")
    print("  " + "-" * 30)
    for _, row in router_curve.iloc[::5].iterrows():
        print(f"  {row['lambda']:>8.1f} {row['accuracy']:>6.3f} ${row['cost']:>9.6f}")

    # Per-benchmark breakdown
    print("\n[6b] Per-benchmark breakdown...")
    benchmarks = df_test["benchmark"].unique()
    for bench in sorted(benchmarks):
        df_bench = df_test[df_test["benchmark"] == bench]
        df_train_bench = df_train[df_train["benchmark"] == bench]
        cs_bench = compute_cluster_stats_minimal(df_train_bench)

        print(f"\n  --- {bench} ({df_bench['prompt_id'].nunique()} test prompts) ---")
        gpt54_bench = evaluate_fixed_strategy(df_bench, "gpt-5.4", 0.0)
        print(f"  GPT-5.4:  acc={gpt54_bench['accuracy']:.3f}  cost=${gpt54_bench['cost']:.6f}")

        for lam in [0, 1, 10, 100, 500]:
            r = evaluate_router(df_bench, cs_bench, lam)
            if r["count"] > 0:
                print(f"  {'router lambda=' + str(lam):<20s} acc={r['accuracy']:.3f}  cost=${r['cost']:.6f}")

    # ===== OUT-OF-DOMAIN: FinanceBench (live inference) =====
    print("\n[6c] FinanceBench out-of-domain evaluation (live)...")

    # Load KMeans centroids for cluster assignment
    centroids_path = os.path.join(str(RESULTS_DIR), "centroids.npy")
    centroids = np.load(centroids_path)
    from sklearn.cluster import KMeans
    n_clusters = len(centroids)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.cluster_centers_ = centroids
    kmeans._n_threads = 1

    fb_results = await evaluate_financebench(cluster_stats, kmeans)

    # Save results
    print("\n[7] Saving evaluation results...")
    eval_results = {
        "baselines": {
            "gpt54_only": gpt54_baseline,
            "openrouter": openrouter_baseline,
        },
        "router_curve": router_curve.to_dict("records"),
        "no_compress_curve": no_compress_curve.to_dict("records"),
        "auc_router": auc_router,
        "auc_no_compress": auc_no_compress,
        "qnc": qnc,
        "financebench": fb_results,
    }

    eval_path = os.path.join(str(RESULTS_DIR), "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=str)
    print(f"  Saved to {eval_path}")

    router_curve.to_csv(os.path.join(str(RESULTS_DIR), "deferral_curve.csv"), index=False)
    no_compress_curve.to_csv(os.path.join(str(RESULTS_DIR), "deferral_curve_no_compress.csv"), index=False)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
