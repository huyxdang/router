"""Generate all evaluation plots."""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
GRID_PATH = os.path.join(RESULTS_DIR, "grid_results_clustered.parquet")
EVAL_PATH = os.path.join(RESULTS_DIR, "evaluation.json")

COLORS = {
    "claude-haiku": "#E07B4B",
    "claude-sonnet": "#D4A054",
    "gpt-4o-mini": "#4B9FE0",
    "router": "#2ECC71",
    "router_cheap": "#27AE60",
}

MODEL_MARKERS = {
    "claude-haiku": "s",
    "claude-sonnet": "D",
    "gpt-4o-mini": "o",
}


def load_data():
    df = pd.read_parquet(GRID_PATH)
    with open(EVAL_PATH) as f:
        evaluation = json.load(f)
    return df, evaluation


def plot_1_deferral_curve(df, evaluation):
    """Deferral curve: accuracy vs cost for router and baselines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot baseline points
    baselines = evaluation["baselines"]
    for name, result in baselines.items():
        if result["cost"] == 0:
            continue
        # Parse model name from baseline key
        if "sonnet" in name:
            model = "claude-sonnet"
        elif "haiku" in name:
            model = "claude-haiku"
        else:
            model = "gpt-4o-mini"

        agg = name.split("_")[-1].replace("agg", "").replace("compress", "0.0")
        if agg == "no":
            agg = "0.0"
        else:
            agg = f"0.{agg.replace('0', '')}" if len(agg) == 2 else agg

        ax.scatter(
            result["cost"], result["accuracy"],
            c=COLORS[model], marker=MODEL_MARKERS[model],
            s=100, zorder=5, edgecolors="black", linewidths=0.5,
        )
        ax.annotate(
            f"{name.replace('_', ' ')}",
            (result["cost"], result["accuracy"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=7, alpha=0.8,
        )

    # Plot router deferral curve
    curve = pd.DataFrame(evaluation["router_curve"])
    curve = curve.sort_values("cost").drop_duplicates(subset="cost", keep="last")
    ax.plot(curve["cost"], curve["accuracy"], color=COLORS["router"],
            linewidth=2.5, label="Adaptive Router (all models)", zorder=4)

    # Plot cheap router curve
    curve_cheap = pd.DataFrame(evaluation["router_curve_cheap"])
    curve_cheap = curve_cheap.sort_values("cost").drop_duplicates(subset="cost", keep="last")
    ax.plot(curve_cheap["cost"], curve_cheap["accuracy"], color=COLORS["router_cheap"],
            linewidth=2.5, linestyle="--", label="Adaptive Router (haiku + gpt4o-mini)", zorder=4)

    ax.set_xlabel("Average Cost per Request (USD)", fontsize=12)
    ax.set_ylabel("Accuracy (LLM Judge)", fontsize=12)
    ax.set_title("Deferral Curve: Router vs Fixed Baselines", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_deferral_curve.png"), dpi=150)
    plt.close()
    print("  Saved 01_deferral_curve.png")


def plot_2_compression_vs_accuracy(df):
    """Accuracy vs aggressiveness for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    metrics = [("llm_judge_correct", "Accuracy (LLM Judge)"), ("f1_score", "F1 Score"), ("contains_answer", "Contains Answer")]

    for ax, (metric, title) in zip(axes, metrics):
        for model in df["model_name"].unique():
            sub = df[df["model_name"] == model]
            means = sub.groupby("aggressiveness")[metric].mean()
            ax.plot(means.index, means.values, marker="o", color=COLORS[model],
                    label=model, linewidth=2, markersize=6)

        ax.set_xlabel("Aggressiveness", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        agg_levels = sorted(df["aggressiveness"].unique())
        ax.set_xticks(agg_levels)

    plt.suptitle("Compression Impact on Accuracy by Model", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_compression_vs_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 02_compression_vs_accuracy.png")


def plot_3_cost_vs_accuracy_scatter(df):
    """Scatter: every (model, agg) combo — cost vs F1."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in df["model_name"].unique():
        for agg in df["aggressiveness"].unique():
            sub = df[(df["model_name"] == model) & (df["aggressiveness"] == agg)]
            ax.scatter(
                sub["total_cost_usd"].mean(),
                sub["llm_judge_correct"].mean(),
                c=COLORS[model],
                marker=MODEL_MARKERS[model],
                s=80 + agg * 100,  # bigger = more aggressive
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.annotate(
                f"agg={agg}",
                (sub["total_cost_usd"].mean(), sub["llm_judge_correct"].mean()),
                textcoords="offset points", xytext=(6, 3),
                fontsize=7, alpha=0.7,
            )

    # Legend
    for model in df["model_name"].unique():
        ax.scatter([], [], c=COLORS[model], marker=MODEL_MARKERS[model],
                   s=80, label=model, edgecolors="black", linewidths=0.5)
    ax.legend(fontsize=10)

    ax.set_xlabel("Average Cost per Request (USD)", fontsize=12)
    ax.set_ylabel("Accuracy (LLM Judge)", fontsize=12)
    ax.set_title("Cost vs Accuracy: All (Model, Aggressiveness) Combinations", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_cost_vs_accuracy.png"), dpi=150)
    plt.close()
    print("  Saved 03_cost_vs_accuracy.png")


def plot_4_routing_heatmap(df):
    """Heatmap: best (model, agg) per cluster at different λ values."""
    lambda_values = [0, 1, 10, 100, 500, 1000]

    agg_dict = {"mean_f1": ("f1_score", "mean"), "mean_cost": ("total_cost_usd", "mean")}
    if "llm_judge_correct" in df.columns:
        agg_dict["mean_judge"] = ("llm_judge_correct", "mean")
    cluster_stats = df.groupby(["cluster_id", "model_name", "aggressiveness"]).agg(**agg_dict).reset_index()

    n_clusters = df["cluster_id"].nunique()
    clusters = sorted(df["cluster_id"].unique())

    # Build a matrix: rows=clusters, cols=lambda values
    # Value = encoded (model, agg) choice
    all_models = sorted(df["model_name"].unique())
    all_aggs = sorted(df["aggressiveness"].unique())

    # Create encoding: each unique (model, agg) gets a number
    combos = [(m, a) for m in all_models for a in all_aggs]
    combo_to_idx = {c: i for i, c in enumerate(combos)}

    matrix = np.zeros((n_clusters, len(lambda_values)), dtype=int)
    labels = [[None] * len(lambda_values) for _ in range(n_clusters)]

    for j, lam in enumerate(lambda_values):
        for i, cid in enumerate(clusters):
            candidates = cluster_stats[cluster_stats["cluster_id"] == cid].copy()
            metric_col = "mean_judge" if "mean_judge" in candidates.columns else "mean_f1"
            candidates["score"] = (1 - candidates[metric_col]) + lam * candidates["mean_cost"]
            best = candidates.loc[candidates["score"].idxmin()]
            choice = (best["model_name"], best["aggressiveness"])
            matrix[i, j] = combo_to_idx.get(choice, 0)
            labels[i][j] = f"{best['model_name'][:6]}\n{best['aggressiveness']}"

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="tab20")

    ax.set_xticks(range(len(lambda_values)))
    ax.set_xticklabels([f"λ={l}" for l in lambda_values])
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"C{c}" for c in clusters])

    # Add text labels
    for i in range(n_clusters):
        for j in range(len(lambda_values)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=6)

    ax.set_xlabel("Lambda (cost-quality tradeoff)", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_title("Routing Decisions: Best (Model, Agg) per Cluster at Different λ", fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_routing_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved 04_routing_heatmap.png")


def plot_5_benchmark_comparison(df):
    """Compare accuracy across benchmarks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, benchmark in zip(axes, df["benchmark"].unique()):
        sub = df[df["benchmark"] == benchmark]
        metric_col = "llm_judge_correct" if "llm_judge_correct" in sub.columns else "f1_score"
        pivot = sub.pivot_table(values=metric_col, index="aggressiveness",
                                columns="model_name", aggfunc="mean")
        pivot.plot(kind="bar", ax=ax, color=[COLORS[m] for m in pivot.columns],
                   edgecolor="black", linewidth=0.5)
        ax.set_title(f"{benchmark.upper()}: Accuracy by Model & Aggressiveness", fontsize=12)
        ax.set_xlabel("Aggressiveness", fontsize=11)
        ax.set_ylabel("Accuracy (LLM Judge)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_benchmark_comparison.png"), dpi=150)
    plt.close()
    print("  Saved 05_benchmark_comparison.png")


def plot_6_cost_breakdown(df):
    """Stacked bar: LLM cost vs bear cost by model and aggressiveness."""
    fig, ax = plt.subplots(figsize=(12, 5))

    groups = df.groupby(["model_name", "aggressiveness"]).agg(
        llm_cost=("total_llm_cost_usd", "mean"),
        bear_cost=("bear_cost_usd", "mean"),
    ).reset_index()

    x_labels = [f"{r['model_name'][:6]}\nagg={r['aggressiveness']}" for _, r in groups.iterrows()]
    x = np.arange(len(groups))

    ax.bar(x, groups["llm_cost"] * 1000, label="LLM Cost", color="#4B9FE0", edgecolor="black", linewidth=0.5)
    ax.bar(x, groups["bear_cost"] * 1000, bottom=groups["llm_cost"] * 1000,
           label="Bear Cost", color="#E07B4B", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Avg Cost per Request (× $0.001)", fontsize=11)
    ax.set_title("Cost Breakdown: LLM vs Bear Compression", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_cost_breakdown.png"), dpi=150)
    plt.close()
    print("  Saved 06_cost_breakdown.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    df, evaluation = load_data()
    print(f"  {len(df)} records loaded\n")

    plot_1_deferral_curve(df, evaluation)
    plot_2_compression_vs_accuracy(df)
    plot_3_cost_vs_accuracy_scatter(df)
    plot_4_routing_heatmap(df)
    plot_5_benchmark_comparison(df)
    plot_6_cost_breakdown(df)

    print(f"\n  All plots saved to {PLOTS_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
