"""Tune K (clusters) and |S_val| (profiling set size) via cross-validation.

Sweeps K × |S_val| grid. For each combo:
  1. Cluster train prompts (K clusters)
  2. Sample |S_val| prompts to build cluster stats (profiling)
  3. Evaluate AUC on remaining val prompts (evaluation)
  4. 5-fold CV for robust estimates

Saves best (K, |S_val|) to tuning_best.json for 06_build_router.py.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    K_VALUES, RESULTS_DIR, RANDOM_SEED,
    TRAIN_FRACTION, VAL_FRACTION,
)
from router.data import load_prompts
from router.embeddings import embed_and_cache
from router.clustering import compute_cluster_stats_minimal
from router.scoring import evaluate_router, compute_auc, LAMBDA_VALUES

GRID_PATH = os.path.join(str(RESULTS_DIR), "grid_results_judged.parquet")
TUNING_RESULTS_PATH = os.path.join(str(RESULTS_DIR), "tuning_results.json")
TUNING_BEST_PATH = os.path.join(str(RESULTS_DIR), "tuning_best.json")

N_FOLDS = 5
SVAL_SIZES = [25, 50, 100, 200, 500]
LAMBDA_SAMPLES = [0, 10, 100, 500]


def split_prompt_ids(prompt_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split prompt IDs into train/val/test (deterministic)."""
    rng = np.random.RandomState(RANDOM_SEED)
    ids = prompt_ids.copy()
    rng.shuffle(ids)

    n_train = int(len(ids) * TRAIN_FRACTION)
    n_val = int(len(ids) * VAL_FRACTION)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def run_fold(df, train_ids, eval_ids, embeddings_map, k, profile_ids):
    """Run one fold: cluster on train, profile on profile_ids, evaluate on eval_ids."""
    df_profile = df[df["prompt_id"].isin(profile_ids)]
    df_eval = df[df["prompt_id"].isin(eval_ids)]

    # Cluster using train prompt embeddings
    train_embeddings = np.array([embeddings_map[pid] for pid in train_ids])
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    kmeans.fit(train_embeddings)

    # Assign clusters to profile prompts
    profile_embeddings = np.array([embeddings_map[pid] for pid in profile_ids])
    profile_labels = kmeans.predict(profile_embeddings)
    profile_cluster_map = dict(zip(profile_ids, profile_labels))
    df_profile = df_profile.copy()
    df_profile["cluster_id"] = df_profile["prompt_id"].map(profile_cluster_map)

    # Assign clusters to eval prompts
    eval_embeddings = np.array([embeddings_map[pid] for pid in eval_ids])
    eval_labels = kmeans.predict(eval_embeddings)
    eval_cluster_map = dict(zip(eval_ids, eval_labels))
    df_eval = df_eval.copy()
    df_eval["cluster_id"] = df_eval["prompt_id"].map(eval_cluster_map)

    # Build cluster stats from profile set only
    cluster_stats = compute_cluster_stats_minimal(df_profile)

    # Compute AUC from full deferral curve
    curve_points = []
    for lam in LAMBDA_VALUES:
        r = evaluate_router(df_eval, cluster_stats, lam)
        curve_points.append({"cost": r["cost"], "accuracy": r["accuracy"]})
    auc = compute_auc(pd.DataFrame(curve_points))

    # Sample lambda results for reporting
    fold_results = {}
    for lam in LAMBDA_SAMPLES:
        fold_results[lam] = evaluate_router(df_eval, cluster_stats, lam)
    fold_results["auc"] = auc

    return fold_results


def main():
    print("=" * 80)
    print("TUNING: K x |S_val| SWEEP")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_parquet(GRID_PATH)
    df["llm_judge_correct"] = (df["llm_judge"] == "correct").astype(float)
    print(f"  {len(df)} records, {df['prompt_id'].nunique()} prompts")

    # Embed
    print("\n[2] Embedding prompts...")
    prompts = load_prompts()
    prompt_ids = [p["id"] for p in prompts]
    prompt_texts = [p["text"] for p in prompts]
    embeddings = embed_and_cache(prompt_ids, prompt_texts)
    embeddings_map = dict(zip(prompt_ids, embeddings))

    # Split — only tune on val set (test is held out entirely)
    print("\n[3] Splitting prompts...")
    train_ids, val_ids, test_ids = split_prompt_ids(prompt_ids)
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Only val prompts that appear in the grid results
    val_ids_in_grid = [pid for pid in val_ids if pid in df["prompt_id"].values]
    print(f"  Val prompts in grid: {len(val_ids_in_grid)}")

    # Filter |S_val| sizes to those <= available val prompts
    max_sval = len(val_ids_in_grid)
    sval_sizes = [s for s in SVAL_SIZES if s <= max_sval]
    if max_sval not in sval_sizes:
        sval_sizes.append(max_sval)
    print(f"  |S_val| sizes to sweep: {sval_sizes}")

    # Create CV folds over val set
    print(f"\n[4] Creating {N_FOLDS}-fold CV over val set...")
    rng = np.random.RandomState(RANDOM_SEED)
    val_shuffled = val_ids_in_grid.copy()
    rng.shuffle(val_shuffled)
    folds = np.array_split(val_shuffled, N_FOLDS)
    print(f"  Fold sizes: {[len(f) for f in folds]}")

    # Sweep K × |S_val|
    total_combos = len(K_VALUES) * len(sval_sizes)
    print(f"\n[5] Sweeping {len(K_VALUES)} K values x {len(sval_sizes)} |S_val| sizes "
          f"= {total_combos} combos x {N_FOLDS} folds")

    all_results = {}
    best_k = None
    best_sval = None
    best_auc = -1

    for k in K_VALUES:
        for sval in sval_sizes:
            key = f"K={k}_Sval={sval}"
            fold_aucs = []
            fold_details = []

            for fold_idx in range(N_FOLDS):
                eval_ids = list(folds[fold_idx])
                remaining = [pid for i, f in enumerate(folds) for pid in f if i != fold_idx]

                # Sample |S_val| from remaining val prompts for profiling
                profile_rng = np.random.RandomState(RANDOM_SEED + fold_idx)
                n_profile = min(sval, len(remaining))
                profile_ids = list(profile_rng.choice(remaining, size=n_profile, replace=False))

                result = run_fold(df, train_ids, eval_ids, embeddings_map, k, profile_ids)
                fold_aucs.append(result["auc"])
                fold_details.append(result)

            auc_mean = float(np.mean(fold_aucs))
            auc_std = float(np.std(fold_aucs))

            summary = {
                "k": k,
                "sval": sval,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
            }
            for lam in LAMBDA_SAMPLES:
                accs = [fd[lam]["accuracy"] for fd in fold_details]
                costs = [fd[lam]["cost"] for fd in fold_details]
                summary[f"lambda={lam}_acc_mean"] = float(np.mean(accs))
                summary[f"lambda={lam}_acc_std"] = float(np.std(accs))
                summary[f"lambda={lam}_cost_mean"] = float(np.mean(costs))

            all_results[key] = summary

            marker = ""
            if auc_mean > best_auc:
                best_auc = auc_mean
                best_k = k
                best_sval = sval
                marker = " <-- best"

            print(f"  {key:<20s}  AUC={auc_mean:.6f}+/-{auc_std:.6f}{marker}")

    # Final comparison
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print("=" * 80)

    # Print table header
    header = f"  {'K':>3s} {'|S_val|':>7s} {'AUC':>12s}"
    for lam in LAMBDA_SAMPLES:
        header += f"  {'λ='+str(lam)+' acc':>12s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for key in sorted(all_results.keys(), key=lambda x: all_results[x]["auc_mean"], reverse=True):
        s = all_results[key]
        row = f"  {s['k']:>3d} {s['sval']:>7d} {s['auc_mean']:>8.6f}+/-{s['auc_std']:.4f}"
        for lam in LAMBDA_SAMPLES:
            row += f"  {s[f'lambda={lam}_acc_mean']:>6.3f}+/-{s[f'lambda={lam}_acc_std']:.3f}"
        print(row)

    print(f"\n  Best: K={best_k}, |S_val|={best_sval} (AUC={best_auc:.6f})")

    # Save full results
    with open(TUNING_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Full results: {TUNING_RESULTS_PATH}")

    # Save best config for 06_build_router.py
    best_config = {
        "best_k": best_k,
        "best_sval": best_sval,
        "best_auc": best_auc,
    }
    with open(TUNING_BEST_PATH, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"  Best config:  {TUNING_BEST_PATH}")

    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print("=" * 80)
    print(f"  Next: python scripts/06_build_router.py")


if __name__ == "__main__":
    main()
