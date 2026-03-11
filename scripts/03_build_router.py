"""Build the routing table: embed prompts, cluster, compute per-cluster stats.

Uses train-only data for clustering (never sees test prompts).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    K_VALUES, AGGRESSIVENESS_LEVELS, MODELS,
    RESULTS_DIR, TRAIN_FRACTION, VAL_FRACTION, RANDOM_SEED,
)
from router.data import load_prompts
from router.embeddings import embed_and_cache
from router.clustering import compute_cluster_stats

GRID_RESULTS_PATH = os.path.join(str(RESULTS_DIR), "grid_results_judged.parquet")

# Deployable router artifacts
ROUTER_CONFIG_PATH = os.path.join(str(RESULTS_DIR), "router_config.json")
CENTROIDS_PATH = os.path.join(str(RESULTS_DIR), "centroids.npy")
CLUSTER_STATS_PATH = os.path.join(str(RESULTS_DIR), "cluster_stats.parquet")

# Offline-only (evaluation, reproducibility)
SPLITS_PATH = os.path.join(str(RESULTS_DIR), "router_splits.json")
EMBEDDINGS_PATH = os.path.join(str(RESULTS_DIR), "router_embeddings.npz")


def split_prompt_ids(prompt_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split prompt IDs into train/val/test."""
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

    # Step 1: Load grid results
    print("\n[1] Loading grid results...")
    if not os.path.exists(GRID_RESULTS_PATH):
        raw_path = os.path.join(str(RESULTS_DIR), "grid_results.parquet")
        msg = [
            f"Missing required file: {GRID_RESULTS_PATH}",
            "03_build_router.py expects judged results (column: llm_judge).",
        ]
        if os.path.exists(raw_path):
            msg.append(f"Found raw grid results at: {raw_path}")
            msg.append("Next step: run judge batch before building router:")
            msg.append("  python3 scripts/03_llm_judge_batch.py submit")
            msg.append("  python3 scripts/03_llm_judge_batch.py status")
            msg.append("  python3 scripts/03_llm_judge_batch.py download")
        raise SystemExit("\n".join(msg))

    df = pd.read_parquet(GRID_RESULTS_PATH)
    if "llm_judge" not in df.columns:
        raise SystemExit(
            f"{GRID_RESULTS_PATH} is missing column 'llm_judge'. "
            "Re-run: python3 scripts/03_llm_judge_batch.py download"
        )
    df["llm_judge_correct"] = (df["llm_judge"] == "correct").astype(float)
    print(f"  {len(df)} records")

    # Step 2: Load prompts and embed
    print("\n[2] Embedding prompts...")
    prompts = load_prompts()
    prompt_ids = [p["id"] for p in prompts]
    prompt_texts = [p["text"] for p in prompts]

    embeddings = embed_and_cache(prompt_ids, prompt_texts)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Step 3: Train/val/test split — cluster on TRAIN only
    print("\n[3] Splitting prompts (train/val/test)...")
    train_ids, val_ids, test_ids = split_prompt_ids(prompt_ids)
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    id_to_idx = {pid: i for i, pid in enumerate(prompt_ids)}
    train_embeddings = np.array([embeddings[id_to_idx[pid]] for pid in train_ids])

    # Step 4: K-means clustering — sweep K values
    print(f"\n[4] Clustering with K values: {K_VALUES}")

    for k in K_VALUES:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(train_embeddings)
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  K={k:3d}: min_cluster={counts.min()}, max={counts.max()}, "
              f"mean={counts.mean():.1f}")

    # Use middle K as default; final selection happens in evaluation
    mid_idx = len(K_VALUES) // 2
    default_k = K_VALUES[mid_idx]
    print(f"\n  Default K={default_k} (final selection by AUC in evaluation)")
    kmeans = KMeans(n_clusters=default_k, random_state=RANDOM_SEED, n_init=10)
    kmeans.fit(train_embeddings)

    # Assign clusters to ALL prompts (train via fit, val/test via predict)
    all_embeddings_ordered = np.array([embeddings[id_to_idx[pid]] for pid in prompt_ids])
    cluster_labels = kmeans.predict(all_embeddings_ordered)
    prompt_to_cluster = dict(zip(prompt_ids, [int(c) for c in cluster_labels]))

    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"  Cluster sizes: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.1f}, median={np.median(counts):.0f}")

    # Step 5: Assign cluster IDs to grid results
    print("\n[5] Assigning cluster IDs to grid results...")
    df["cluster_id"] = df["prompt_id"].map(prompt_to_cluster)
    missing = df["cluster_id"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} records have no cluster assignment")
    df["cluster_id"] = df["cluster_id"].astype(int)

    # Step 6: Compute per-cluster stats (from train+val only, not test)
    print("\n[6] Computing per-cluster statistics (train+val only)...")
    train_val_ids = set(train_ids + val_ids)
    df_train_val = df[df["prompt_id"].isin(train_val_ids)]
    cluster_stats = compute_cluster_stats(df_train_val)

    n_combos = len(cluster_stats)
    print(f"  {n_combos} (cluster, model, agg) combinations")
    print(f"  Expected: {default_k} x {len(MODELS)} x {len(AGGRESSIVENESS_LEVELS)} = "
          f"{default_k * len(MODELS) * len(AGGRESSIVENESS_LEVELS)}")

    # Step 7: Quick sanity check
    print("\n[7] Best (model, agg) per cluster (by judge accuracy):")
    for cid in sorted(cluster_stats["cluster_id"].unique()):
        sub = cluster_stats[cluster_stats["cluster_id"] == cid]
        best = sub.loc[sub["mean_judge"].idxmax()]
        print(f"  Cluster {cid:2d} ({int(best['count']):2d} prompts): "
              f"{best['model_name']:<16s} agg={best['aggressiveness']:.1f}  "
              f"judge={best['mean_judge']:.3f}  cost=${best['mean_cost']:.5f}")

    # Step 8: Save everything (portable formats, no pickle)
    print("\n[8] Saving router artifacts...")

    # Deployable: config JSON
    router_config = {
        "models_available": [m["name"] for m in MODELS],
        "agg_levels": AGGRESSIVENESS_LEVELS,
        "n_clusters": default_k,
    }
    with open(ROUTER_CONFIG_PATH, "w") as f:
        json.dump(router_config, f, indent=2)
    print(f"  Config:        {ROUTER_CONFIG_PATH}")

    # Deployable: KMeans centroids
    np.save(CENTROIDS_PATH, kmeans.cluster_centers_)
    print(f"  Centroids:     {CENTROIDS_PATH}  ({kmeans.cluster_centers_.shape})")

    # Deployable: cluster stats
    cluster_stats.to_parquet(CLUSTER_STATS_PATH, index=False)
    print(f"  Cluster stats: {CLUSTER_STATS_PATH}")

    # Offline: train/val/test splits + prompt mapping
    splits = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "prompt_ids": prompt_ids,
        "prompt_to_cluster": {k: int(v) for k, v in prompt_to_cluster.items()},
    }
    with open(SPLITS_PATH, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"  Splits:        {SPLITS_PATH}")

    # Offline: embeddings
    np.savez_compressed(EMBEDDINGS_PATH, embeddings=embeddings)
    print(f"  Embeddings:    {EMBEDDINGS_PATH}")

    # Also save enriched grid results with cluster IDs
    enriched_path = os.path.join(str(RESULTS_DIR), "grid_results_clustered.parquet")
    df.to_parquet(enriched_path, index=False)
    print(f"  Enriched results saved to {enriched_path}")

    print("\n" + "=" * 80)
    print("ROUTER BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
