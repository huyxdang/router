# TTC Work Trial: Adaptive Compression Router

## The Pitch

UniRoute (Google, 2025) solves model routing with dynamic LLM pools using cluster-based prompt representations. But UniRoute only routes across models — it always sends the full prompt. We extend UniRoute to **jointly route across models AND compression levels**, using TTC's bear API as both the compression engine and the feature extractor. This is a novel contribution: compression-aware routing with theoretical grounding from UniRoute's framework.

---

## Architecture

```
User constraints (models, budget, agg range)
          ↓
Prompt → Embed → Nearest Cluster → Filter to user's models → Score valid (model, agg) pairs → Best option → Compress → LLM → Output
```

The routing rule (adapted from UniRoute eq. 9, with user constraints):

```
route(x, user_models, budget, agg_range) =

    argmin over (model m, aggressiveness a):
        error(cluster(x), m, a) + λ * cost(m, a)

    subject to:
        m ∈ user_models                              # only models the user selected
        cost(m, a) ≤ budget.max_cost_per_request      # hard budget ceiling
        agg_range.min ≤ a ≤ agg_range.max             # user's compression bounds

    where cost(m, a) = model_cost_per_token(m) * output_tokens(x, a)
```

λ controls the cost-quality tradeoff. Sweep λ from 0 → ∞ to trace a deferral curve.

The key design advantage: because the routing table stores per-cluster stats for ALL models and ALL aggressiveness levels, filtering to any user-defined subset is just a lookup-time filter. No retraining needed when users pick different model pools.

### User-Facing API Design

```python
response = ttc.smart_route(
    input=prompt,

    # Required: user picks their model pool
    models=[
        {"name": "claude-sonnet", "api_key": "...", "cost_per_1m_input": 3.00},
        {"name": "claude-haiku", "api_key": "...", "cost_per_1m_input": 1.00},
    ],

    # Optional: budget constraints
    budget={
        "max_cost_per_request": 0.005,       # hard ceiling per call in USD
        # OR
        "max_input_cost_per_1m": 5.0,        # max $/1M input tokens
    },

    # Optional: compression bounds
    compression={
        "min_aggressiveness": 0.0,           # never compress less than this
        "max_aggressiveness": 0.6,           # never compress more than this
        # OR
        "auto": True,                        # let router decide within full range
    },

    # Optional: optimization preference
    # Default: balanced (optimize accuracy, cost, and latency jointly)
    optimize="balanced",                     # or "cost" or "quality" or "speed"
)
```

Three modes of operation:
- **Full auto** (default): router optimizes all three metrics jointly within user's model pool
- **Budget-constrained**: router maximizes accuracy while staying under budget
- **Manual override**: user sets fixed aggressiveness + fixed model, router is bypassed

---

## Day 1: Data Collection + Grid Search 

### Step 1: Setup & Dependencies (30 min)

```
- Python environment
- TTC bear API access (confirm key works)
- LLM API keys (Claude Haiku 4.5, Claude Sonnet 4.6, Mistral Large 3)
- sentence-transformers for embeddings
- scikit-learn for K-means
- Benchmark datasets: SQuAD 2.0, CoQA (use their existing benchmark subsets if available, else sample 150 questions each)
```

### Step 2: Prepare Benchmark Data (1 hour)

```python
# For each benchmark, need:
# - prompt (the question + context)
# - ground_truth (the correct answer)
# - evaluation function (exact match, F1, etc.)

benchmarks = {
    "squad2": load_squad2_subset(n=150),
    "coqa": load_coqa_subset(n=150),
}
```

### Step 3: Embed All Prompts (30 min)

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # fast, good enough
embeddings = embedder.encode(all_prompts)  # shape: (N, 384)
```

### Step 4: K-Means Clustering (30 min)

```python
from sklearn.cluster import KMeans

# Try K = 10, 20, 50 — tune later
kmeans = KMeans(n_clusters=20, random_state=42)
cluster_assignments = kmeans.fit_predict(embeddings)
```

### Step 5: The Big Grid Search

This is the expensive step. For each prompt, run every (model, aggressiveness) combination.

```python
aggressiveness_levels = [0.1, 0.4, 0.7, 1.0]
models = ["claude-haiku", "claude-sonnet", "mistral-large"]

results = []

for prompt in benchmark_prompts:
    for agg in aggressiveness_levels:
        # Step A: Compress
        compressed = ttc_compress(prompt.text, aggressiveness=agg)

        for model in models:
            # Step B: Run LLM
            start = time.time()
            output = call_llm(model, compressed.output_text)
            latency = time.time() - start

            # Step C: Evaluate
            results.append({
                "prompt_id": prompt.id,
                "cluster_id": cluster_assignments[prompt.idx],
                "benchmark": prompt.benchmark,
                "aggressiveness": agg,
                "model": model,
                "accuracy": evaluate(output, prompt.ground_truth),
                "original_tokens": compressed.original_input_tokens,
                "output_tokens": compressed.output_tokens,
                "compression_ratio": compressed.output_tokens / compressed.original_input_tokens,
                "cost": compute_cost(model, compressed.output_tokens),
                "latency": latency,
            })
```

**IMPORTANT: Budget the API costs.**
- 150 prompts × 2 benchmarks × 4 agg levels × 3 models = 3,600 LLM calls
- At ~500 tokens avg input, this is ~3.15M tokens total
- Estimate cost before running. If too expensive, reduce to:
  - 100 prompts per benchmark
  - 4 agg levels: [0.1, 0.4, 0.7, 1.0]
  - 2 models

**Parallelization:** Use async requests to speed this up. Batch the bear API calls.

### Step 6: Compute Per-Cluster Statistics (1 hour)

This is the core of UniRoute — represent each (model, aggressiveness) pair by its per-cluster error rate. We store stats for ALL models so any user-defined subset works at inference time.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(results)

# For each (cluster, model, aggressiveness), compute:
# - mean accuracy
# - mean cost
# - mean latency
cluster_stats = df.groupby(["cluster_id", "model", "aggressiveness"]).agg({
    "accuracy": "mean",
    "cost": "mean",
    "latency": "mean",
    "compression_ratio": "mean",
}).reset_index()

# IMPORTANT: Do NOT precompute a fixed routing table.
# Instead, store cluster_stats and compute the optimal decision at inference time
# based on the user's constraints.
# This enables dynamic model pools — any subset of models works without retraining.
```

**Inference-time routing function (handles all user constraints):**

```python
def route(prompt_embedding, cluster_stats, user_config):
    """
    Route a prompt to the optimal (model, aggressiveness) given user constraints.
    
    user_config = {
        "models": ["claude-sonnet", "claude-haiku"],       # user's chosen models
        "max_cost_per_request": 0.005,                   # budget ceiling (optional)
        "min_aggressiveness": 0.0,                       # compression floor (optional)
        "max_aggressiveness": 0.6,                       # compression ceiling (optional)
        "lambda": 1.0,                                   # cost-quality tradeoff
    }
    """
    # 1. Find nearest cluster
    cluster_id = kmeans.predict(prompt_embedding.reshape(1, -1))[0]
    
    # 2. Get all stats for this cluster
    cluster_data = cluster_stats[cluster_stats.cluster_id == cluster_id].copy()
    
    # 3. Filter to user's chosen models
    cluster_data = cluster_data[cluster_data["model"].isin(user_config["models"])]
    
    # 4. Filter to user's aggressiveness range
    min_agg = user_config.get("min_aggressiveness", 0.0)
    max_agg = user_config.get("max_aggressiveness", 0.6)
    cluster_data = cluster_data[
        (cluster_data["aggressiveness"] >= min_agg) &
        (cluster_data["aggressiveness"] <= max_agg)
    ]
    
    # 5. Filter by budget constraint (if set)
    if "max_cost_per_request" in user_config:
        cluster_data = cluster_data[
            cluster_data["cost"] <= user_config["max_cost_per_request"]
        ]
    
    # 6. If no valid options remain, relax constraints and warn
    if len(cluster_data) == 0:
        return {"error": "No valid (model, aggressiveness) pair satisfies all constraints"}
    
    # 7. Score remaining options: minimize error + λ * cost
    lam = user_config.get("lambda", 1.0)
    cluster_data["score"] = (1 - cluster_data["accuracy"]) + lam * cluster_data["cost"]
    
    # 8. Pick the best
    best_idx = cluster_data["score"].idxmin()
    best = cluster_data.loc[best_idx]
    
    return {
        "model": best["model"],
        "aggressiveness": best["aggressiveness"],
        "expected_accuracy": best["accuracy"],
        "expected_cost": best["cost"],
        "cluster_id": cluster_id,
    }
```

**Why this works for dynamic model pools:**

The cluster_stats table has entries for ALL models in the grid search. When user A picks [sonnet, haiku] and user B picks [mistral-large, haiku], they both use the same underlying table — the filtering happens at inference time. Adding a new model only requires running it on 150 validation prompts to compute its per-cluster stats, then appending those rows to cluster_stats. No retraining.

### Step 7: Save Everything (15 min)

```python
# Save for Day 2
pickle.dump({
    "kmeans": kmeans,
    "embedder_name": "all-MiniLM-L6-v2",
    "cluster_stats": cluster_stats,    # raw stats — routing computed at inference time
    "raw_results": df,
    "models_available": models,        # all models we have stats for
    "agg_levels": aggressiveness_levels,
}, open("router_data.pkl", "wb"))
```

---

## Day 2: Evaluation + Demo 

### Step 8: Evaluate the Router 

Hold out 20% of prompts from the grid search. Compare routing strategies:

```
Baselines to compare:
1. No compression, always best model (Sonnet)         → upper bound on accuracy
2. No compression, always cheapest model (Mistral Large) → lower bound on cost
3. Fixed 0.4 aggressiveness, always Sonnet             → bear compression alone
4. Fixed 0.4 aggressiveness, always Haiku              → cheap baseline
5. Your adaptive router                                → joint optimization
```

**Evaluation metrics (from UniRoute):**

```python
# 1. Deferral curve: sweep λ, plot accuracy vs. cost
#    - For each λ, compute average accuracy and average cost of routed decisions
#    - Plot as a curve

# 2. Area under deferral curve (AUC)
#    - Higher = better tradeoff

# 3. Quality-Neutral Cost (QNC)
#    - Minimum cost to match the accuracy of the best model (no compression)
#    - Lower = better
```

### Step 9: Analysis & Visualizations

Create the following plots:

```
Plot 1: Deferral curve
  - X axis: average cost per prompt
  - Y axis: average accuracy
  - Lines: each baseline + your router
  - Individual LLM points plotted as markers
  → This is the money chart. Your router's curve should dominate baselines.

Plot 2: Routing decisions heatmap
  - X axis: cluster ID
  - Y axis: λ value
  - Color: which (model, aggressiveness) was chosen
  → Shows the router making different decisions per cluster per budget

Plot 3: Compression ratio vs. accuracy by model
  - One line per model
  - Shows each model's "breaking point" under compression
  → Insight: cheap models break earlier, so adaptive compression matters more for them

Plot 4: Cumulative cost comparison over N turns
  - Simulate a 20-turn conversation
  - Compare: no compression vs. fixed vs. adaptive
  - Show total $ spent
  → This is the business case chart
```

### Step 10: Build the Demo

Streamlit app or Jupyter notebook with interactive demo:

```python
import streamlit as st

st.title("TTC Adaptive Compression Router")
st.caption("UniRoute-inspired joint compression + model routing")

# ============ User Constraint Panel ============
st.sidebar.header("Configuration")

# Model selection — user picks from available models
all_models = {
    "claude-haiku": {"cost_per_1m_input": 1.00, "cost_per_1m_output": 5.00},
    "claude-sonnet": {"cost_per_1m_input": 3.00, "cost_per_1m_output": 15.00},
    "mistral-large": {"cost_per_1m_input": 0.50, "cost_per_1m_output": 1.50},
}
selected_models = st.sidebar.multiselect(
    "Select models:",
    list(all_models.keys()),
    default=["claude-haiku", "claude-sonnet"],
)

# Budget constraint
budget_enabled = st.sidebar.checkbox("Set budget ceiling")
max_cost = None
if budget_enabled:
    max_cost = st.sidebar.number_input(
        "Max cost per request ($)", 
        min_value=0.0001, max_value=0.1, value=0.005, step=0.001, format="%.4f"
    )

# Compression bounds
st.sidebar.subheader("Compression bounds")
agg_range = st.sidebar.slider(
    "Aggressiveness range",
    min_value=0.0, max_value=0.6, value=(0.0, 0.6), step=0.1,
)

# Cost-quality tradeoff
optimize = st.sidebar.slider("Cost ↔ Quality tradeoff (λ)", 0.0, 50.0, 1.0)

# ============ Main Panel ============
prompt = st.text_area("Enter prompt:")

if st.button("Route & Compress"):
    if not selected_models:
        st.error("Select at least one model.")
    else:
        # Build user config
        user_config = {
            "models": selected_models,
            "min_aggressiveness": agg_range[0],
            "max_aggressiveness": agg_range[1],
            "lambda": optimize,
        }
        if max_cost:
            user_config["max_cost_per_request"] = max_cost

        # 1. Embed prompt
        embedding = embedder.encode([prompt])

        # 2. Route with user constraints
        decision = route(embedding, cluster_stats, user_config)

        if "error" in decision:
            st.error(decision["error"])
            st.info("Try: adding more models, increasing budget, or widening compression range.")
        else:
            # 3. Compress
            compressed = ttc_compress(prompt, aggressiveness=decision["aggressiveness"])

            # 4. Call LLM
            response = call_llm(decision["model"], compressed.output_text)

            # Display results
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model", decision["model"])
            col2.metric("Aggressiveness", f"{decision['aggressiveness']:.1f}")
            col3.metric("Tokens saved", f"{(1-compressed.ratio)*100:.0f}%")
            col4.metric("Est. cost", f"${decision['expected_cost']:.4f}")

            st.write("### Response")
            st.write(response)

            # ============ Comparison Panel ============
            st.write("### What other configs would have done")
            
            # Show all valid options for this cluster, ranked by score
            cluster_id = decision["cluster_id"]
            options = cluster_stats[cluster_stats.cluster_id == cluster_id].copy()
            options = options[options["model"].isin(selected_models)]
            options["score"] = (1 - options["accuracy"]) + optimize * options["cost"]
            options = options.sort_values("score")
            
            st.dataframe(options[["model", "aggressiveness", "accuracy", "cost", "score"]].head(10))
```

### Step 11: Write the Summary (1 hour)

Short writeup (1-2 pages) covering:

```
1. Problem: Fixed compression + fixed model is suboptimal. Different prompts
   tolerate different compression levels, and different models handle compression
   differently. Users also have different model preferences and budgets.

2. Approach: Extend UniRoute's cluster-based routing to jointly optimize
   compression aggressiveness and model selection. Use K-means prompt clustering
   with sentence embeddings, compute per-cluster (model, aggressiveness)
   performance grid, route via cost-adjusted error minimization. Users define
   constraints (model pool, budget, compression range) and the router optimizes
   within those constraints. New models can be added with just 150 validation
   calls — no retraining required.

3. Results:
   - Deferral curve showing router dominates fixed baselines
   - QNC showing X% cost reduction at equivalent accuracy
   - Specific examples of adaptive routing decisions
   - Demonstration of dynamic model pool support

4. What this enables for TTC:
   - New product feature: auto-aggressiveness (remove the manual parameter)
   - Model-aware compression (different models need different compression levels)
   - Dynamic model pools (users pick any subset of models, router adapts)
   - Budget-constrained optimization (stay under $/request ceiling)
   - Foundation for agent-to-agent compression routing

5. Future work / production path:
   - Replace K-means with learned cluster map (UniRoute LearnedMap)
   - Use bear's own token-level confidence as additional features
   - GRPO-based online learning from production API usage
   - Extend to conversation history compression (sliding aggressiveness)
   - Per-customer learned λ based on usage patterns
```

---

## File Structure

```
ttc-adaptive-router/
├── README.md                   # Summary writeup
├── requirements.txt
├── config.py                   # API keys, model configs, costs
├── data/
│   ├── squad2_subset.json
│   └── coqa_subset.json
├── scripts/
│   ├── 01_embed_and_cluster.py
│   ├── 02_grid_search.py       # The big experiment
│   ├── 03_build_routing_table.py
│   ├── 04_evaluate.py          # Baselines vs. router
│   └── 05_visualize.py         # Generate all plots
├── router/
│   ├── __init__.py
│   ├── compress.py             # TTC API wrapper
│   ├── llm.py                  # LLM API wrapper (multi-model)
│   ├── router.py               # The actual router logic
│   └── evaluate.py             # Evaluation metrics (deferral curves, QNC)
├── demo/
│   └── app.py                  # Streamlit demo
├── results/
│   ├── grid_results.csv
│   ├── routing_table.pkl
│   └── plots/
│       ├── deferral_curve.png
│       ├── routing_heatmap.png
│       ├── compression_vs_accuracy.png
│       └── cumulative_cost.png
└── notebooks/
    └── analysis.ipynb          # Exploratory analysis
```

---

## Risk Mitigation

**Risk: API costs blow up.**
Mitigation: Start with 50 prompts × 2 models × 4 agg levels = 400 calls. Validate the pipeline works, then scale up.

**Risk: Grid search takes too long.**
Mitigation: Use async requests. Batch bear API calls. Run overnight if needed.

**Risk: All models perform the same under compression.**
Mitigation: This would actually be an interesting finding — document it. If true, the value is purely in auto-aggressiveness, not model routing.

**Risk: Not enough signal to differentiate clusters.**
Mitigation: Try different K values. Also try bear-derived features (compression ratio at fixed aggressiveness) as additional clustering features alongside sentence embeddings.