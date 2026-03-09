"""Evaluate the router against baselines: deferral curves, AUC, QNC."""

from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MODELS, AGGRESSIVENESS_LEVELS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
ROUTER_PATH = os.path.join(RESULTS_DIR, "router.pkl")
GRID_PATH = os.path.join(RESULTS_DIR, "grid_results_clustered.parquet")

# Hold out 20% of prompts for evaluation
TEST_FRACTION = 0.2
RANDOM_SEED = 42

# Lambda values to sweep for deferral curves
LAMBDA_VALUES = np.concatenate([
    np.arange(0, 10, 0.5),
    np.arange(10, 100, 5),
    np.arange(100, 1001, 50),
])


def split_prompts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test by prompt_id."""
    prompt_ids = df["prompt_id"].unique()
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(prompt_ids)

    n_test = max(1, int(len(prompt_ids) * TEST_FRACTION))
    test_ids = set(prompt_ids[:n_test])
    train_ids = set(prompt_ids[n_test:])

    df_train = df[df["prompt_id"].isin(train_ids)]
    df_test = df[df["prompt_id"].isin(test_ids)]

    print(f"  Train: {len(train_ids)} prompts ({len(df_train)} records)")
    print(f"  Test:  {len(test_ids)} prompts ({len(df_test)} records)")
    return df_train, df_test


def build_cluster_stats(df_train: pd.DataFrame) -> pd.DataFrame:
    """Build cluster stats from training data only."""
    return df_train.groupby(["cluster_id", "model_name", "aggressiveness"]).agg(
        mean_f1=("f1_score", "mean"),
        mean_ca=("contains_answer", "mean"),
        mean_em=("correct", "mean"),
        mean_judge=("llm_judge_correct", "mean"),
        mean_cost=("total_cost_usd", "mean"),
    ).reset_index()


def evaluate_fixed_strategy(df_test: pd.DataFrame, model_name: str,
                            aggressiveness: float, metric: str = "f1_score") -> dict:
    """Evaluate a fixed (model, agg) strategy on test data."""
    sub = df_test[
        (df_test["model_name"] == model_name) &
        (df_test["aggressiveness"] == aggressiveness)
    ]
    if len(sub) == 0:
        return {"accuracy": 0.0, "cost": 0.0, "count": 0}

    return {
        "accuracy": sub[metric].mean(),
        "cost": sub["total_cost_usd"].mean(),
        "count": len(sub),
    }


def evaluate_router(df_test: pd.DataFrame, cluster_stats: pd.DataFrame,
                    lambda_: float, metric: str = "f1_score",
                    models: list[str] | None = None) -> dict:
    """Evaluate the adaptive router at a given λ on test data."""
    metric_col = {"f1_score": "mean_f1", "contains_answer": "mean_ca", "correct": "mean_em", "llm_judge_correct": "mean_judge"}[metric]

    if models is None:
        models = cluster_stats["model_name"].unique().tolist()

    # For each test prompt, the router picks the best (model, agg) for its cluster
    accuracies = []
    costs = []

    for prompt_id in df_test["prompt_id"].unique():
        prompt_rows = df_test[df_test["prompt_id"] == prompt_id]
        cluster_id = prompt_rows["cluster_id"].iloc[0]

        # Get candidates for this cluster
        candidates = cluster_stats[
            (cluster_stats["cluster_id"] == cluster_id) &
            (cluster_stats["model_name"].isin(models))
        ].copy()

        if len(candidates) == 0:
            continue

        # Score and pick best
        candidates["score"] = (1 - candidates[metric_col]) + lambda_ * candidates["mean_cost"]
        best = candidates.loc[candidates["score"].idxmin()]
        chosen_model = best["model_name"]
        chosen_agg = best["aggressiveness"]

        # Look up actual test performance for this prompt with that (model, agg)
        actual = prompt_rows[
            (prompt_rows["model_name"] == chosen_model) &
            (prompt_rows["aggressiveness"] == chosen_agg)
        ]
        if len(actual) == 0:
            continue

        accuracies.append(actual[metric].iloc[0])
        costs.append(actual["total_cost_usd"].iloc[0])

    return {
        "accuracy": np.mean(accuracies) if accuracies else 0.0,
        "cost": np.mean(costs) if costs else 0.0,
        "count": len(accuracies),
    }


def compute_deferral_curve(df_test, cluster_stats, metric, models=None):
    """Sweep λ to trace accuracy vs cost curve."""
    points = []
    for lam in LAMBDA_VALUES:
        result = evaluate_router(df_test, cluster_stats, lam, metric, models)
        points.append({
            "lambda": lam,
            "accuracy": result["accuracy"],
            "cost": result["cost"],
        })
    return pd.DataFrame(points)


def compute_auc(curve: pd.DataFrame) -> float:
    """Area under the deferral curve (accuracy vs cost)."""
    # Sort by cost ascending
    curve = curve.sort_values("cost")
    # Remove duplicate cost points
    curve = curve.drop_duplicates(subset="cost", keep="last")
    if len(curve) < 2:
        return 0.0
    return float(np.trapz(curve["accuracy"], curve["cost"]))


def compute_qnc(curve: pd.DataFrame, target_accuracy: float) -> float | None:
    """Quality-Neutral Cost: minimum cost to reach target accuracy."""
    curve = curve.sort_values("cost")
    above = curve[curve["accuracy"] >= target_accuracy]
    if len(above) == 0:
        return None
    return float(above["cost"].iloc[0])


def main():
    print("=" * 80)
    print("ROUTER EVALUATION")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_parquet(GRID_PATH)
    print(f"  {len(df)} records, {df['prompt_id'].nunique()} prompts")

    # Train/test split
    print("\n[2] Train/test split...")
    df_train, df_test = split_prompts(df)

    # Build cluster stats from TRAIN only
    print("\n[3] Building cluster stats from train set...")
    cluster_stats = build_cluster_stats(df_train)
    print(f"  {len(cluster_stats)} (cluster, model, agg) combos")

    metric = "llm_judge_correct"
    print(f"\n  Evaluation metric: {metric} (LLM-as-judge)")

    # Baselines
    print("\n[4] Evaluating baselines...")
    baselines = {
        "sonnet_no_compress": ("claude-sonnet", 0.0),
        "haiku_no_compress": ("claude-haiku", 0.0),
        "gpt4omini_no_compress": ("gpt-4o-mini", 0.0),
        "sonnet_agg03": ("claude-sonnet", 0.3),
        "haiku_agg03": ("claude-haiku", 0.3),
        "gpt4omini_agg03": ("gpt-4o-mini", 0.3),
        "sonnet_agg06": ("claude-sonnet", 0.6),
        "haiku_agg06": ("claude-haiku", 0.6),
        "gpt4omini_agg06": ("gpt-4o-mini", 0.6),
        "sonnet_agg09": ("claude-sonnet", 0.9),
        "gpt4omini_agg09": ("gpt-4o-mini", 0.9),
    }

    print(f"\n  {'Strategy':<28s} {'F1':>6s} {'Cost':>10s}")
    print("  " + "-" * 48)

    baseline_results = {}
    for name, (model, agg) in baselines.items():
        result = evaluate_fixed_strategy(df_test, model, agg, metric)
        baseline_results[name] = result
        print(f"  {name:<28s} {result['accuracy']:>6.3f} ${result['cost']:>9.6f}")

    # Router deferral curve
    print("\n[5] Computing router deferral curve...")
    router_curve = compute_deferral_curve(df_test, cluster_stats, metric)

    # Deferral curve for router restricted to cheaper models only
    router_curve_cheap = compute_deferral_curve(
        df_test, cluster_stats, metric, models=["claude-haiku", "gpt-4o-mini"]
    )

    # AUC
    print("\n[6] Results...")
    auc_router = compute_auc(router_curve)
    auc_router_cheap = compute_auc(router_curve_cheap)
    print(f"  Router AUC (all models):    {auc_router:.6f}")
    print(f"  Router AUC (cheap models):  {auc_router_cheap:.6f}")

    # QNC — minimum cost to match best no-compression baseline
    best_no_compress = max(
        baseline_results["sonnet_no_compress"]["accuracy"],
        baseline_results["gpt4omini_no_compress"]["accuracy"],
    )
    qnc = compute_qnc(router_curve, best_no_compress)
    best_baseline_cost = baseline_results["sonnet_no_compress"]["cost"]

    print(f"\n  Best no-compression F1:     {best_no_compress:.3f}")
    if qnc is not None:
        print(f"  QNC (router cost to match): ${qnc:.6f}")
        print(f"  Best model cost (sonnet):   ${best_baseline_cost:.6f}")
        if best_baseline_cost > 0:
            savings = (1 - qnc / best_baseline_cost) * 100
            print(f"  Cost reduction:             {savings:.1f}%")
    else:
        print(f"  QNC: router cannot match baseline accuracy")

    # Print router curve summary
    print(f"\n  Router deferral curve (sampled):")
    print(f"  {'λ':>8s} {'F1':>6s} {'Cost':>10s} {'Model picked':>20s}")
    print("  " + "-" * 50)
    for _, row in router_curve.iloc[::5].iterrows():
        # Find what the router picks at this λ
        sample = evaluate_router(df_test, cluster_stats, row["lambda"], metric)
        print(f"  {row['lambda']:>8.1f} {row['accuracy']:>6.3f} ${row['cost']:>9.6f}")

    # Save results
    print("\n[7] Saving evaluation results...")
    eval_results = {
        "baselines": baseline_results,
        "router_curve": router_curve.to_dict("records"),
        "router_curve_cheap": router_curve_cheap.to_dict("records"),
        "auc_router": auc_router,
        "auc_router_cheap": auc_router_cheap,
        "qnc": qnc,
        "best_no_compress_f1": best_no_compress,
        "metric": metric,
    }

    eval_path = os.path.join(RESULTS_DIR, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=str)
    print(f"  Saved to {eval_path}")

    router_curve.to_csv(os.path.join(RESULTS_DIR, "deferral_curve.csv"), index=False)
    router_curve_cheap.to_csv(os.path.join(RESULTS_DIR, "deferral_curve_cheap.csv"), index=False)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
