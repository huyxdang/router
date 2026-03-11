"""Scoring, deferral curves, and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Lambda sweep values for deferral curves
LAMBDA_VALUES = np.concatenate([
    np.arange(0, 10, 0.5),
    np.arange(10, 100, 5),
    np.arange(100, 1001, 50),
])


def score_candidates(candidates: pd.DataFrame, lambda_: float) -> pd.Series:
    """Score Virtual Model candidates: minimize error + λ * cost.

    Returns the best row from candidates.
    """
    candidates = candidates.copy()
    candidates["score"] = (1 - candidates["mean_judge"]) + lambda_ * candidates["mean_cost"]
    return candidates.loc[candidates["score"].idxmin()]


def evaluate_router(df_test: pd.DataFrame, cluster_stats: pd.DataFrame,
                    lambda_: float,
                    models: list[str] | None = None,
                    agg_filter: list[float] | None = None) -> dict:
    """Evaluate the adaptive router at a given lambda on test data.

    Args:
        df_test: Test set with columns: prompt_id, cluster_id, model_name,
                 aggressiveness, llm_judge_correct, total_cost_usd
        cluster_stats: Per-cluster stats from compute_cluster_stats.
        lambda_: Cost-quality tradeoff parameter.
        models: Restrict to these model names.
        agg_filter: Restrict to these aggressiveness levels (e.g. [0.0] for no-compression).
    """
    if models is None:
        models = cluster_stats["model_name"].unique().tolist()

    accuracies = []
    costs = []

    for prompt_id in df_test["prompt_id"].unique():
        prompt_rows = df_test[df_test["prompt_id"] == prompt_id]
        cluster_id = prompt_rows["cluster_id"].iloc[0]

        candidates = cluster_stats[
            (cluster_stats["cluster_id"] == cluster_id) &
            (cluster_stats["model_name"].isin(models))
        ].copy()

        if agg_filter is not None:
            candidates = candidates[candidates["aggressiveness"].isin(agg_filter)]

        if len(candidates) == 0:
            continue

        best = score_candidates(candidates, lambda_)
        chosen_model = best["model_name"]
        chosen_agg = best["aggressiveness"]

        actual = prompt_rows[
            (prompt_rows["model_name"] == chosen_model) &
            (prompt_rows["aggressiveness"] == chosen_agg)
        ]
        if len(actual) == 0:
            continue

        accuracies.append(actual["llm_judge_correct"].iloc[0])
        costs.append(actual["total_cost_usd"].iloc[0])

    return {
        "accuracy": np.mean(accuracies) if accuracies else 0.0,
        "cost": np.mean(costs) if costs else 0.0,
        "count": len(accuracies),
    }


def compute_deferral_curve(df_test: pd.DataFrame, cluster_stats: pd.DataFrame,
                           models: list[str] | None = None,
                           agg_filter: list[float] | None = None,
                           lambda_values: np.ndarray | None = None) -> pd.DataFrame:
    """Sweep lambda to trace accuracy vs cost curve."""
    if lambda_values is None:
        lambda_values = LAMBDA_VALUES

    points = []
    for lam in lambda_values:
        result = evaluate_router(df_test, cluster_stats, lam, models, agg_filter)
        points.append({
            "lambda": lam,
            "accuracy": result["accuracy"],
            "cost": result["cost"],
        })
    return pd.DataFrame(points)


def compute_auc(curve: pd.DataFrame) -> float:
    """Area under the deferral curve (accuracy vs cost)."""
    curve = curve.sort_values("cost")
    curve = curve.drop_duplicates(subset="cost", keep="last")
    if len(curve) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(curve["accuracy"], curve["cost"])
    else:
        area = np.trapz(curve["accuracy"], curve["cost"])
    return float(area)


def compute_qnc(curve: pd.DataFrame, target_accuracy: float) -> float | None:
    """Quality-Neutral Cost: minimum cost to reach target accuracy."""
    curve = curve.sort_values("cost")
    above = curve[curve["accuracy"] >= target_accuracy]
    if len(above) == 0:
        return None
    return float(above["cost"].iloc[0])
