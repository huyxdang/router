"""Evaluate the router against baselines: deferral curves, AUC, QNC.

Uses Modal judge (Qwen2.5-7B) for consistency with grid search judging.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MODELS, RESULTS_DIR, DATA_DIR,
    EVAL_BENCHMARKS, SYSTEM_PROMPTS, DEFAULT_SYSTEM_PROMPT,
    BATCH_SIZE, get_model_by_name, OPENROUTER_MODEL_IDS,
)
from router.data import load_prompts, load_ground_truths
from router.clustering import compute_cluster_stats_minimal
from router.scoring import (
    evaluate_router, compute_deferral_curve, compute_auc, compute_qnc,
    score_candidates,
)
from router.evaluate import compute_cost
from router.embeddings import embed_and_cache
from router.compress import compress_async
from router.llm import call_llm_async
from router.judge import judge_responses

_api_semaphore = None

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

    Sends each test prompt to OpenRouter, then judges via Modal for consistency.
    """
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

    # Phase 1: Call OpenRouter for all test prompts
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

    # Phase 2: Judge via Modal (consistent with grid search)
    print(f"    Judging {len(or_results)} OpenRouter responses via Modal...")
    gt_list = [ground_truths.get(r["prompt_id"], "") for r in or_results]
    resp_list = [r["response_text"] for r in or_results]
    verdicts = judge_responses(gt_list, resp_list)

    correct = sum(1 for v in verdicts if v == "correct")
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

    Embed -> cluster -> pick (model, agg) -> compress -> call LLM -> judge via Modal.
    """
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

    results_by_lambda = {}

    for lam in lambda_values:
        # Determine routing decisions for each prompt
        routing_decisions = []
        for i, prompt in enumerate(fb_prompts):
            cluster_id = int(fb_clusters[i])
            candidates = cluster_stats[
                cluster_stats["cluster_id"] == cluster_id
            ].copy()

            if len(candidates) == 0:
                routing_decisions.append(None)
                continue

            best = score_candidates(candidates, lam)
            routing_decisions.append({
                "model_name": best["model_name"],
                "aggressiveness": float(best["aggressiveness"]),
            })

        # Run compress + LLM for each prompt (async, semaphore-throttled)
        async def _run_one(i, prompt, decision):
            if decision is None:
                return None
            async with _api_semaphore:
                try:
                    comp = await compress_async(prompt["text"], decision["aggressiveness"])
                except Exception as e:
                    print(f"    Compress error {prompt['id']}: {e}")
                    return None

                model_config = get_model_by_name(decision["model_name"])
                sys_prompt = SYSTEM_PROMPTS.get(prompt["benchmark"], DEFAULT_SYSTEM_PROMPT)
                try:
                    llm_result = await call_llm_async(model_config, comp["compressed_text"], sys_prompt)
                except Exception as e:
                    print(f"    LLM error {prompt['id']}: {e}")
                    return None

            cost_info = compute_cost(
                model_config,
                llm_result["input_tokens"],
                llm_result["output_tokens"],
                comp["tokens_removed"],
            )
            return {
                "ground_truth": prompt["ground_truth"],
                "response_text": llm_result["response_text"],
                "cost": cost_info["total_cost_usd"],
            }

        tasks = [_run_one(i, p, d) for i, (p, d) in enumerate(zip(fb_prompts, routing_decisions))]
        raw_results = await asyncio.gather(*tasks)
        valid_results = [r for r in raw_results if r is not None]

        if not valid_results:
            results_by_lambda[lam] = {"accuracy": 0.0, "cost": 0.0, "count": 0}
            continue

        # Judge all responses via Modal in one batch
        print(f"    Judging {len(valid_results)} FinanceBench responses (lambda={lam})...")
        gt_list = [r["ground_truth"] for r in valid_results]
        resp_list = [r["response_text"] for r in valid_results]
        verdicts = judge_responses(gt_list, resp_list)

        correct = sum(1 for v in verdicts if v == "correct")
        total = len(valid_results)
        total_cost = sum(r["cost"] for r in valid_results)

        results_by_lambda[lam] = {
            "accuracy": correct / total if total else 0.0,
            "cost": total_cost / total if total else 0.0,
            "count": total,
        }
        print(f"    lambda={lam}: acc={results_by_lambda[lam]['accuracy']:.3f}  "
              f"cost=${results_by_lambda[lam]['cost']:.6f}  n={total}")

    return results_by_lambda


async def main():
    global _api_semaphore
    _api_semaphore = asyncio.Semaphore(BATCH_SIZE)

    print("=" * 80)
    print("ROUTER EVALUATION")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    if not os.path.exists(GRID_PATH):
        # Fall back to judged results without cluster IDs
        fallback = os.path.join(str(RESULTS_DIR), "grid_results_judged.parquet")
        if os.path.exists(fallback):
            print(f"  {GRID_PATH} not found, using {fallback}")
            df = pd.read_parquet(fallback)
        else:
            raise SystemExit(f"Missing grid results. Run 06_build_router.py first.")
    else:
        df = pd.read_parquet(GRID_PATH)

    df["llm_judge_correct"] = (df["llm_judge"] == "correct").astype(float)
    print(f"  {len(df)} records, {df['prompt_id'].nunique()} prompts")

    # Load splits
    print("\n[2] Loading splits...")
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    test_ids = set(splits["test_ids"])
    profile_ids = set(splits.get("profile_ids", splits["val_ids"]))

    df_profile = df[df["prompt_id"].isin(profile_ids)]
    df_test = df[df["prompt_id"].isin(test_ids)]
    print(f"  Profile: {df_profile['prompt_id'].nunique()} prompts ({len(df_profile)} records)")
    print(f"  Test:    {df_test['prompt_id'].nunique()} prompts ({len(df_test)} records)")

    # Build cluster stats from profile set (matches 06_build_router.py)
    print("\n[3] Building cluster stats from profile set...")
    cluster_stats = compute_cluster_stats_minimal(df_profile)
    print(f"  {len(cluster_stats)} (cluster, model, agg) combos")

    # ===== BASELINE 1: GPT-5.4 Only =====
    print("\n[4] Evaluating baselines...")

    print("\n  --- Baseline 1: GPT-5.4 Only (no router, no compression) ---")
    gpt54_baseline = evaluate_fixed_strategy(df_test, "gpt-5.4", 0.0)
    print(f"  GPT-5.4 acc={gpt54_baseline['accuracy']:.3f}  cost=${gpt54_baseline['cost']:.6f}")

    # ===== BASELINE 2: OpenRouter =====
    print("\n  --- Baseline 2: OpenRouter (same pool, no compression) ---")
    openrouter_baseline = await evaluate_openrouter_baseline(df_test)
    print(f"  OpenRouter acc={openrouter_baseline['accuracy']:.3f}  "
          f"cost=${openrouter_baseline['cost']:.6f}  "
          f"n={openrouter_baseline['count']}")

    # ===== BASELINE 3: UniRoute No Compression =====
    print("\n  --- Baseline 3: UniRoute No Compression (agg=0.0 only) ---")
    no_compress_curve = compute_deferral_curve(
        df_test, cluster_stats, models=None, agg_filter=[0.0]
    )
    auc_no_compress = compute_auc(no_compress_curve)
    nc_quality = evaluate_router(df_test, cluster_stats, 0.0, agg_filter=[0.0])
    print(f"  No-compress (lambda=0) acc={nc_quality['accuracy']:.3f}  "
          f"cost=${nc_quality['cost']:.6f}")
    print(f"  No-compress AUC={auc_no_compress:.6f}")

    # ===== OUR ROUTER =====
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
        df_profile_bench = df_profile[df_profile["benchmark"] == bench]
        cs_bench = compute_cluster_stats_minimal(df_profile_bench)

        print(f"\n  --- {bench} ({df_bench['prompt_id'].nunique()} test prompts) ---")
        gpt54_bench = evaluate_fixed_strategy(df_bench, "gpt-5.4", 0.0)
        print(f"  GPT-5.4:  acc={gpt54_bench['accuracy']:.3f}  cost=${gpt54_bench['cost']:.6f}")

        for lam in [0, 1, 10, 100, 500]:
            r = evaluate_router(df_bench, cs_bench, lam)
            if r["count"] > 0:
                print(f"  {'router lambda=' + str(lam):<20s} acc={r['accuracy']:.3f}  cost=${r['cost']:.6f}")

    # ===== OUT-OF-DOMAIN: FinanceBench =====
    print("\n[6c] FinanceBench out-of-domain evaluation (live)...")

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
