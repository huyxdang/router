"""Build the final router using best (K, |S_val|) from tuning.

Reads tuning_best.json from 05_tune.py, then:
  1. Clusters train prompts with best K
  2. Samples best |S_val| from val set for profiling (cluster stats)
  3. Saves deployable router artifacts (JSON + npy + parquet)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    AGGRESSIVENESS_LEVELS, MODELS,
    RESULTS_DIR, TRAIN_FRACTION, VAL_FRACTION, RANDOM_SEED,
)
from router.data import load_prompts
from router.embeddings import embed_and_cache
from router.clustering import compute_cluster_stats

GRID_RESULTS_PATH = os.path.join(str(RESULTS_DIR), "grid_results_judged.parquet")
TUNING_BEST_PATH = os.path.join(str(RESULTS_DIR), "tuning_best.json")

# Deployable router artifacts
ROUTER_CONFIG_PATH = os.path.join(str(RESULTS_DIR), "router_config.json")
CENTROIDS_PATH = os.path.join(str(RESULTS_DIR), "centroids.npy")
CLUSTER_STATS_PATH = os.path.join(str(RESULTS_DIR), "cluster_stats.parquet")

# Offline-only (evaluation, reproducibility)
SPLITS_PATH = os.path.join(str(RESULTS_DIR), "router_splits.json")
EMBEDDINGS_PATH = os.path.join(str(RESULTS_DIR), "router_embeddings.npz")


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


def main():
    os.makedirs(str(RESULTS_DIR), exist_ok=True)

    print("=" * 80)
    print("BUILD ROUTER")
    print("=" * 80)

    # Step 1: Load tuning results
    print("\n[1] Loading tuning config...")
    if not os.path.exists(TUNING_BEST_PATH):
        raise SystemExit(
            f"Missing {TUNING_BEST_PATH}. Run 05_tune.py first."
        )
    with open(TUNING_BEST_PATH) as f:
        tuning = json.load(f)

    best_k = tuning["best_k"]
    best_sval = tuning["best_sval"]
    print(f"  Best K={best_k}, |S_val|={best_sval} (AUC={tuning['best_auc']:.6f})")

    # Step 2: Load grid results
    print("\n[2] Loading grid results...")
    if not os.path.exists(GRID_RESULTS_PATH):
        raise SystemExit(f"Missing {GRID_RESULTS_PATH}. Run grid search + judge first.")
    df = pd.read_parquet(GRID_RESULTS_PATH)
    df["llm_judge_correct"] = (df["llm_judge"] == "correct").astype(float)
    print(f"  {len(df)} records")

    # Step 3: Embed
    print("\n[3] Embedding prompts...")
    prompts = load_prompts()
    prompt_ids = [p["id"] for p in prompts]
    prompt_texts = [p["text"] for p in prompts]
    embeddings = embed_and_cache(prompt_ids, prompt_texts)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Step 4: Split
    print("\n[4] Splitting prompts...")
    train_ids, val_ids, test_ids = split_prompt_ids(prompt_ids)
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    id_to_idx = {pid: i for i, pid in enumerate(prompt_ids)}

    # Step 5: Cluster on train with best K
    print(f"\n[5] Clustering (K={best_k})...")
    train_embeddings = np.array([embeddings[id_to_idx[pid]] for pid in train_ids])
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    kmeans.fit(train_embeddings)

    # Assign clusters to all prompts
    all_embeddings = np.array([embeddings[id_to_idx[pid]] for pid in prompt_ids])
    cluster_labels = kmeans.predict(all_embeddings)
    prompt_to_cluster = dict(zip(prompt_ids, [int(c) for c in cluster_labels]))

    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"  Cluster sizes: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.1f}")

    # Step 6: Assign clusters to grid results
    print("\n[6] Assigning cluster IDs...")
    df["cluster_id"] = df["prompt_id"].map(prompt_to_cluster)
    missing = df["cluster_id"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} records have no cluster assignment")
    df["cluster_id"] = df["cluster_id"].astype(int)

    # Step 7: Sample |S_val| from val for profiling, compute cluster stats
    print(f"\n[7] Profiling with |S_val|={best_sval}...")
    val_ids_in_grid = [pid for pid in val_ids if pid in df["prompt_id"].values]
    rng = np.random.RandomState(RANDOM_SEED)

    if best_sval >= len(val_ids_in_grid):
        profile_ids = val_ids_in_grid
    else:
        profile_ids = list(rng.choice(val_ids_in_grid, size=best_sval, replace=False))
    print(f"  Using {len(profile_ids)} val prompts for profiling")

    df_profile = df[df["prompt_id"].isin(profile_ids)]
    cluster_stats = compute_cluster_stats(df_profile)

    n_combos = len(cluster_stats)
    print(f"  {n_combos} (cluster, model, agg) combinations")

    # Step 8: Sanity check
    print(f"\n[8] Best (model, agg) per cluster:")
    for cid in sorted(cluster_stats["cluster_id"].unique()):
        sub = cluster_stats[cluster_stats["cluster_id"] == cid]
        best = sub.loc[sub["mean_judge"].idxmax()]
        print(f"  Cluster {cid:2d} ({int(best['count']):2d} prompts): "
              f"{best['model_name']:<16s} agg={best['aggressiveness']:.1f}  "
              f"judge={best['mean_judge']:.3f}  cost=${best['mean_cost']:.5f}")

    # Step 9: Save artifacts
    print(f"\n[9] Saving router artifacts...")

    router_config = {
        "models_available": [m["name"] for m in MODELS],
        "agg_levels": AGGRESSIVENESS_LEVELS,
        "n_clusters": best_k,
        "sval_size": best_sval,
        "tuning_auc": tuning["best_auc"],
    }
    with open(ROUTER_CONFIG_PATH, "w") as f:
        json.dump(router_config, f, indent=2)
    print(f"  Config:        {ROUTER_CONFIG_PATH}")

    np.save(CENTROIDS_PATH, kmeans.cluster_centers_)
    print(f"  Centroids:     {CENTROIDS_PATH}  ({kmeans.cluster_centers_.shape})")

    cluster_stats.to_parquet(CLUSTER_STATS_PATH, index=False)
    print(f"  Cluster stats: {CLUSTER_STATS_PATH}")

    splits = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "profile_ids": profile_ids,
        "prompt_ids": prompt_ids,
        "prompt_to_cluster": {k: int(v) for k, v in prompt_to_cluster.items()},
    }
    with open(SPLITS_PATH, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"  Splits:        {SPLITS_PATH}")

    np.savez_compressed(EMBEDDINGS_PATH, embeddings=embeddings)
    print(f"  Embeddings:    {EMBEDDINGS_PATH}")

    enriched_path = os.path.join(str(RESULTS_DIR), "grid_results_clustered.parquet")
    df.to_parquet(enriched_path, index=False)
    print(f"  Enriched:      {enriched_path}")

    print("\n" + "=" * 80)
    print("ROUTER BUILD COMPLETE")
    print("=" * 80)
    print(f"  K={best_k}, |S_val|={best_sval}")
    print(f"  Next: python scripts/07_evaluate.py")


if __name__ == "__main__":
    main()
