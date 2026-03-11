"""Adaptive compression router: embed -> cluster -> route to optimal (model, agg)."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from router.embeddings import embed_single, embed_texts
from router.scoring import score_candidates


class Router:
    """Routes prompts to optimal (model, aggressiveness) pairs based on cluster stats.

    Loads from portable artifacts (JSON + npy + parquet), not pickle.
    """

    def __init__(self, router_dir: str):
        """Load router from a directory containing router_config.json, centroids.npy,
        and cluster_stats.parquet.
        """
        # Load config
        with open(os.path.join(router_dir, "router_config.json")) as f:
            config = json.load(f)
        self.models_available = config["models_available"]
        self.agg_levels = config["agg_levels"]
        n_clusters = config["n_clusters"]

        # Reconstruct KMeans from centroids
        centroids = np.load(os.path.join(router_dir, "centroids.npy"))
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.kmeans.cluster_centers_ = centroids
        self.kmeans._n_threads = 1

        # Load cluster stats
        self.cluster_stats = pd.read_parquet(
            os.path.join(router_dir, "cluster_stats.parquet")
        )

    def route(self, prompt: str, user_config: dict | None = None) -> dict:
        """Route a prompt to the optimal (model, aggressiveness).

        user_config options:
            models: list[str]           — restrict to these models
            max_cost_per_request: float  — hard budget ceiling
            min_aggressiveness: float    — compression floor
            max_aggressiveness: float    — compression ceiling
            lambda_: float              — cost-quality tradeoff (default 1.0)
        """
        if user_config is None:
            user_config = {}

        embedding = embed_single(prompt)
        cluster_id = int(self.kmeans.predict(embedding.reshape(1, -1))[0])
        return self._route_by_cluster(cluster_id, user_config)

    def route_batch(self, prompts: list[str], user_config: dict | None = None) -> list[dict]:
        """Route multiple prompts."""
        if user_config is None:
            user_config = {}

        embeddings = embed_texts(prompts)
        cluster_ids = self.kmeans.predict(embeddings)

        return [
            self._route_by_cluster(int(cid), user_config)
            for cid in cluster_ids
        ]

    def _route_by_cluster(self, cluster_id: int, user_config: dict) -> dict:
        """Route given a known cluster ID (avoids re-embedding)."""
        candidates = self.cluster_stats[
            self.cluster_stats["cluster_id"] == cluster_id
        ].copy()

        # Filter by user constraints
        models = user_config.get("models", self.models_available)
        candidates = candidates[candidates["model_name"].isin(models)]

        min_agg = user_config.get("min_aggressiveness", min(self.agg_levels))
        max_agg = user_config.get("max_aggressiveness", max(self.agg_levels))
        candidates = candidates[
            (candidates["aggressiveness"] >= min_agg) &
            (candidates["aggressiveness"] <= max_agg)
        ]

        if "max_cost_per_request" in user_config:
            candidates = candidates[
                candidates["mean_cost"] <= user_config["max_cost_per_request"]
            ]

        if len(candidates) == 0:
            return {"error": "No valid (model, aggressiveness) pair satisfies all constraints"}

        lam = user_config.get("lambda_", 1.0)
        best = score_candidates(candidates, lam)

        return {
            "model": best["model_name"],
            "aggressiveness": float(best["aggressiveness"]),
            "expected_accuracy": float(best["mean_judge"]),
            "expected_cost": float(best["mean_cost"]),
            "expected_latency": float(best["mean_latency"]),
            "cluster_id": cluster_id,
            "score": float(best["score"]),
        }
