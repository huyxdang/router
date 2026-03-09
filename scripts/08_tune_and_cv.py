"""K tuning + cross-validation: find optimal number of clusters with robust evaluation."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EMBEDDING_MODEL, MODELS, AGGRESSIVENESS_LEVELS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
GRID_PATH = os.path.join(RESULTS_DIR, "grid_results_judged.parquet")

K_VALUES = [5, 10, 15, 20, 30]
N_FOLDS = 5
RANDOM_SEED = 42
METRIC = "llm_judge_correct"

# Representative λ values: quality-focused, balanced, cost-focused
LAMBDA_SAMPLES = [0, 10, 100, 500]

LAMBDA_CURVE = np.concatenate([
    np.arange(0, 10, 0.5),
    np.arange(10, 100, 5),
    np.arange(100, 1001, 50),
])


def load_prompts():
    prompts = []
    for filename in ["squad2_subset.json", "finqa_subset.json"]:
        with open(os.path.join(DATA_DIR, filename)) as f:
            prompts.extend(json.load(f))
    return prompts


def build_cluster_stats(df_train):
    return df_train.groupby(["cluster_id", "model_name", "aggressiveness"]).agg(
        mean_judge=("llm_judge_correct", "mean"),
        mean_cost=("total_cost_usd", "mean"),
    ).reset_index()


def evaluate_router_at_lambda(df_test, cluster_stats, lambda_):
    metric_col = "mean_judge"
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

        if len(candidates) == 0:
            continue

        candidates["score"] = (1 - candidates[metric_col]) + lambda_ * candidates["mean_cost"]
        best = candidates.loc[candidates["score"].idxmin()]

        actual = prompt_rows[
            (prompt_rows["model_name"] == best["model_name"]) &
            (prompt_rows["aggressiveness"] == best["aggressiveness"])
        ]
        if len(actual) == 0:
            continue

        accuracies.append(actual[METRIC].iloc[0])
        costs.append(actual["total_cost_usd"].iloc[0])

    return {
        "accuracy": np.mean(accuracies) if accuracies else 0.0,
        "cost": np.mean(costs) if costs else 0.0,
    }


def compute_auc(curve):
    curve = curve.sort_values("cost").drop_duplicates(subset="cost", keep="last")
    if len(curve) < 2:
        return 0.0
    return float(np.trapezoid(curve["accuracy"], curve["cost"]))


def run_fold(df, prompt_ids, train_ids, test_ids, embeddings_map, k):
    """Run one fold: cluster train prompts, evaluate on test."""
    df_train = df[df["prompt_id"].isin(train_ids)]
    df_test = df[df["prompt_id"].isin(test_ids)]

    # Cluster using train prompt embeddings
    train_embeddings = np.array([embeddings_map[pid] for pid in train_ids])
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    kmeans.fit(train_embeddings)

    # Assign clusters to train
    train_cluster_map = dict(zip(train_ids, kmeans.labels_))
    df_train = df_train.copy()
    df_train["cluster_id"] = df_train["prompt_id"].map(train_cluster_map)

    # Assign clusters to test (predict nearest)
    test_embeddings = np.array([embeddings_map[pid] for pid in test_ids])
    test_labels = kmeans.predict(test_embeddings)
    test_cluster_map = dict(zip(test_ids, test_labels))
    df_test = df_test.copy()
    df_test["cluster_id"] = df_test["prompt_id"].map(test_cluster_map)

    # Build cluster stats from train
    cluster_stats = build_cluster_stats(df_train)

    # Evaluate at sample λ values
    fold_results = {}
    for lam in LAMBDA_SAMPLES:
        result = evaluate_router_at_lambda(df_test, cluster_stats, lam)
        fold_results[lam] = result

    # Compute AUC from full deferral curve
    curve_points = []
    for lam in LAMBDA_CURVE:
        r = evaluate_router_at_lambda(df_test, cluster_stats, lam)
        curve_points.append({"cost": r["cost"], "accuracy": r["accuracy"]})
    auc = compute_auc(pd.DataFrame(curve_points))
    fold_results["auc"] = auc

    return fold_results


def main():
    print("=" * 80)
    print("K TUNING + CROSS-VALIDATION")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_parquet(GRID_PATH)
    df["llm_judge_correct"] = (df["llm_judge"] == "correct").astype(float)
    print(f"  {len(df)} records, {df['prompt_id'].nunique()} prompts")

    # Embed all prompts
    print(f"\n[2] Embedding prompts with {EMBEDDING_MODEL}...")
    prompts = load_prompts()
    prompt_texts = {p["id"]: p["text"] for p in prompts}
    prompt_ids = list(prompt_texts.keys())

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [prompt_texts[pid] for pid in prompt_ids]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings_map = dict(zip(prompt_ids, embeddings))

    # Create CV folds
    print(f"\n[3] Creating {N_FOLDS}-fold CV splits...")
    rng = np.random.RandomState(RANDOM_SEED)
    shuffled = prompt_ids.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, N_FOLDS)
    print(f"  Fold sizes: {[len(f) for f in folds]}")

    # Sweep K values
    print(f"\n[4] Sweeping K = {K_VALUES}...")
    all_results = {}

    for k in K_VALUES:
        print(f"\n  {'='*60}")
        print(f"  K = {k}")
        print(f"  {'='*60}")

        fold_results = []
        for fold_idx in range(N_FOLDS):
            test_ids = list(folds[fold_idx])
            train_ids = [pid for i, f in enumerate(folds) for pid in f if i != fold_idx]

            result = run_fold(df, prompt_ids, train_ids, test_ids, embeddings_map, k)
            fold_results.append(result)

            # Print fold summary
            lam0 = result[0]
            print(f"    Fold {fold_idx+1}: λ=0 acc={lam0['accuracy']:.3f} | AUC={result['auc']:.6f}")

        # Aggregate across folds
        k_summary = {"k": k}
        for lam in LAMBDA_SAMPLES:
            accs = [fr[lam]["accuracy"] for fr in fold_results]
            costs = [fr[lam]["cost"] for fr in fold_results]
            k_summary[f"λ={lam}_acc_mean"] = np.mean(accs)
            k_summary[f"λ={lam}_acc_std"] = np.std(accs)
            k_summary[f"λ={lam}_cost_mean"] = np.mean(costs)

        aucs = [fr["auc"] for fr in fold_results]
        k_summary["auc_mean"] = np.mean(aucs)
        k_summary["auc_std"] = np.std(aucs)

        all_results[k] = k_summary

        print(f"\n    Summary for K={k}:")
        for lam in LAMBDA_SAMPLES:
            acc_m = k_summary[f"λ={lam}_acc_mean"]
            acc_s = k_summary[f"λ={lam}_acc_std"]
            cost_m = k_summary[f"λ={lam}_cost_mean"]
            print(f"      λ={lam:<4d}  acc={acc_m:.3f}±{acc_s:.3f}  cost=${cost_m:.6f}")
        print(f"      AUC={k_summary['auc_mean']:.6f}±{k_summary['auc_std']:.6f}")

    # Final comparison
    print(f"\n\n{'='*80}")
    print("FINAL COMPARISON")
    print("=" * 80)

    header = f"  {'K':>3s}"
    for lam in LAMBDA_SAMPLES:
        header += f"  {'λ='+str(lam)+' acc':>14s}"
    header += f"  {'AUC':>18s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    best_k = None
    best_auc = -1

    for k in K_VALUES:
        s = all_results[k]
        row = f"  {k:>3d}"
        for lam in LAMBDA_SAMPLES:
            row += f"  {s[f'λ={lam}_acc_mean']:.3f}±{s[f'λ={lam}_acc_std']:.3f}"

        row += f"  {s['auc_mean']:.6f}±{s['auc_std']:.6f}"
        print(row)

        if s["auc_mean"] > best_auc:
            best_auc = s["auc_mean"]
            best_k = k

    print(f"\n  Best K by AUC: K={best_k} (AUC={best_auc:.6f})")

    # Save results
    results_path = os.path.join(RESULTS_DIR, "k_tuning_cv.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved to {results_path}")

    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
