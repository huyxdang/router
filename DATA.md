# Data Collection Specification

## Overview

Every API call in the grid search should log a comprehensive record. This dataset is used for both the cluster-based router (UniRoute approach) and the trained ML router. Collect once, use for everything.

---

## Record Schema

```python
record = {
    # ========================================
    # PROMPT METADATA (computed once per prompt)
    # ========================================
    "prompt_id": "squad2_042",
    "benchmark": "squad2",                    # dataset source
    "prompt_text": "...",                     # raw full text
    "ground_truth": "Paris",                  # correct answer
    "prompt_char_length": 1842,               # len(prompt_text)
    "prompt_word_count": 312,                 # len(prompt_text.split())
    "prompt_token_count_approx": 420,         # rough estimate before compression

    # ========================================
    # EMBEDDING (computed once per prompt, reuse across all combos)
    # ========================================
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding": [0.023, -0.041, ...],        # 384-dim vector, store separately in .npy

    # ========================================
    # BEAR COMPRESSION (one call per prompt × aggressiveness combo)
    # ========================================
    "aggressiveness": 0.3,
    "bear_model": "bear-1.1",
    "original_input_tokens": 842,             # from bear response
    "output_tokens": 589,                     # from bear response
    "compression_ratio": 0.699,               # output_tokens / original_input_tokens
    "tokens_removed": 253,                    # original - output
    "removal_rate": 0.301,                    # 1 - compression_ratio
    "compressed_text": "...",                 # SAVE THIS — avoid re-calling bear

    # ========================================
    # LLM CALL (one call per prompt × aggressiveness × model combo)
    # ========================================
    "model_name": "claude-haiku",             # human-readable
    "model_id": "claude-haiku-4-5-20251001",   # exact API model string
    "model_provider": "anthropic",            # anthropic / openai
    "model_cost_per_1m_input": 0.80,          # USD
    "model_cost_per_1m_output": 4.00,         # USD
    "llm_response": "The capital is Paris.",  # SAVE FULL RESPONSE
    "llm_input_tokens": 589,                  # from API response usage field
    "llm_output_tokens": 45,                  # from API response usage field

    # ========================================
    # EVALUATION (computed post-hoc from response + ground truth)
    # ========================================
    "correct": 1,                             # binary exact match (case-insensitive)
    "f1_score": 0.92,                         # token-level F1 overlap
    "contains_answer": 1,                     # ground_truth substring in response

    # ========================================
    # COST (computed from model pricing + token counts)
    # ========================================
    "input_cost_usd": 0.000471,               # llm_input_tokens * cost_per_1m_input / 1e6
    "output_cost_usd": 0.000180,              # llm_output_tokens * cost_per_1m_output / 1e6
    "total_llm_cost_usd": 0.000651,           # input + output
    "bear_cost_usd": 0.0000127,               # tokens_removed * $0.05 / 1e6
    "total_cost_usd": 0.000664,               # llm cost + bear cost

    # ========================================
    # LATENCY (measured during LLM call)
    # ========================================
    "latency_seconds": 1.34,                  # total wall clock time
    "ttfb_seconds": 0.42,                     # time to first byte (if available)

    # ========================================
    # METADATA
    # ========================================
    "timestamp": "2026-03-09T14:23:01Z",
    "trial_id": "run_001",                    # in case you run multiple experiments
}
```

---

## Grid Dimensions

```
Prompts:           150 per benchmark × 2 benchmarks = 300 prompts
Aggressiveness:    [0.1, 0.4, 0.7, 1.0] = 4 levels
Models:            [claude-haiku, claude-sonnet, mistral-large] = 3 models

Total API calls:
  Bear calls:    300 prompts × 4 agg levels = 1,200
  LLM calls:     300 prompts × 4 agg levels × 3 models = 3,600
  Total:         4,800 API calls
```

### Reduced grid (if budget constrained)

```
Prompts:           100 per benchmark × 2 benchmarks = 200 prompts
Aggressiveness:    [0.0, 0.15, 0.3, 0.45, 0.6] = 5 levels
Models:            [claude-haiku, mistral-large] = 2 models

Total API calls:
  Bear calls:    200 × 5 = 1,000
  LLM calls:     200 × 5 × 2 = 2,000
  Total:         3,000 API calls
```

---

## Storage Format

### Primary results: Parquet

```python
import pandas as pd

# Don't store embeddings or full text in the main table
# Keep it lean for fast analysis
df = pd.DataFrame(all_results)

# Drop large text fields for the analysis table
df_lean = df.drop(columns=["prompt_text", "compressed_text", "llm_response", "embedding"])
df_lean.to_parquet("results/grid_results.parquet", index=False)

# Save full results separately (for re-evaluation)
df.to_parquet("results/grid_results_full.parquet", index=False)
```

### Embeddings: NumPy

```python
import numpy as np

# Shape: (num_prompts, 384)
# Indexed by prompt_id
np.save("results/embeddings.npy", all_embeddings)
```

### Compressed texts: JSON lookup

```python
import json

# Key: (prompt_id, aggressiveness) → compressed text
# So you never need to re-call bear
compressed_cache = {}
for r in all_results:
    key = f"{r['prompt_id']}_agg{r['aggressiveness']}"
    compressed_cache[key] = {
        "compressed_text": r["compressed_text"],
        "original_input_tokens": r["original_input_tokens"],
        "output_tokens": r["output_tokens"],
    }

with open("results/compressed_cache.json", "w") as f:
    json.dump(compressed_cache, f)
```

---

## Data Collection Script Structure

```python
import asyncio
import aiohttp
import time
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ============ CONFIG ============
TTC_API_KEY = "..."
ANTHROPIC_API_KEY = "..."
OPENAI_API_KEY = "..."

MODELS = [
    {"name": "claude-haiku", "id": "claude-haiku-4-5-20251001",
     "provider": "anthropic", "input_cost": 1.00, "output_cost": 5.00},
    {"name": "claude-sonnet", "id": "claude-sonnet-4-6",
     "provider": "anthropic", "input_cost": 3.00, "output_cost": 15.00},
    {"name": "mistral-large", "id": "mistral-large-2512",
     "provider": "mistral", "input_cost": 0.50, "output_cost": 1.50},
]

AGG_LEVELS = [0.1, 0.4, 0.7, 1.0]

# ============ STEP 1: Load benchmarks ============
prompts = load_squad2(n=150) + load_coqa(n=150)

# ============ STEP 2: Embed all prompts (one-time) ============
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode([p["text"] for p in prompts])
np.save("results/embeddings.npy", embeddings)

# ============ STEP 3: Compress all prompts at all agg levels ============
compressed_cache = {}
for prompt in prompts:
    for agg in AGG_LEVELS:
        result = call_bear(prompt["text"], agg)
        key = f"{prompt['id']}_agg{agg}"
        compressed_cache[key] = result
        # Save periodically
    print(f"Compressed prompt {prompt['id']}")

with open("results/compressed_cache.json", "w") as f:
    json.dump(compressed_cache, f)

# ============ STEP 4: Run all LLM calls ============
all_results = []

for prompt in prompts:
    for agg in AGG_LEVELS:
        key = f"{prompt['id']}_agg{agg}"
        compressed = compressed_cache[key]

        for model in MODELS:
            start = time.time()
            response = call_llm(model, compressed["compressed_text"])
            latency = time.time() - start

            record = build_record(prompt, agg, compressed, model, response, latency)
            all_results.append(record)

    # Save every 10 prompts (crash protection)
    if len(all_results) % (len(AGG_LEVELS) * len(MODELS) * 10) == 0:
        pd.DataFrame(all_results).to_parquet("results/grid_results_checkpoint.parquet")
        print(f"Checkpoint saved: {len(all_results)} records")

# ============ STEP 5: Final save ============
df = pd.DataFrame(all_results)
df.to_parquet("results/grid_results.parquet", index=False)
print(f"Done. {len(df)} total records.")
```

---

## Cost Estimate

### Bear API costs

```
Compressed tokens charged = tokens removed
At ~40% removal rate on average:
  300 prompts × 4 agg × ~400 tokens avg × 0.40 removal × $0.05/1M
  = 300 × 4 × 160 × $0.00000005
  = $0.010
  → Negligible
```

### LLM API costs (the real cost)

```
Haiku:         300 × 4 × ~500 tokens input × $1.00/1M  = $0.60
Sonnet:        300 × 4 × ~500 tokens input × $3.00/1M  = $1.80
Mistral Large: 300 × 4 × ~500 tokens input × $0.50/1M  = $0.30

Output tokens (~50 each):
Haiku:         300 × 4 × 50 × $5.00/1M  = $0.30
Sonnet:        300 × 4 × 50 × $15.00/1M = $0.90
Mistral Large: 300 × 4 × 50 × $1.50/1M  = $0.09

Total estimated: ~$4
```

### With reduced grid

```
~$2-3 total
```

---

## Validation Checklist

Before running the full grid, validate with 5 prompts:

```
[ ] Bear API key works
[ ] Bear returns expected fields (output, original_input_tokens, output_tokens)
[ ] Each LLM API works and returns valid responses
[ ] Evaluation function correctly scores responses
[ ] Records are saving properly to parquet
[ ] Compressed text cache is working
[ ] Cost calculation matches expected values
[ ] Embeddings are the right shape
```

---

## What This Dataset Enables

```
Cluster-based router:
  - embeddings → K-means clustering
  - group by (cluster, model, agg) → per-cluster accuracy
  - routing table at various λ values

Trained ML router:
  - features: embedding + prompt_length + compression_ratio + ...
  - labels: optimal (model, agg) per prompt (derived from grid)
  - train: GBT, MLP, or logistic regression

Analysis & visualization:
  - deferral curves (accuracy vs cost at various λ)
  - compression tolerance per model
  - prompt difficulty distribution
  - cluster composition analysis

Adding a new model later:
  - use compressed_cache to skip bear calls
  - only need new LLM calls: 300 × 4 = 1,200 calls
  - recompute routing table or retrain model
```