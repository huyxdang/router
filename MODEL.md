# TTC Work Trial: Adaptive Compression Router

## The Pitch

UniRoute (Google, 2025) solves model routing with dynamic LLM pools using cluster-based prompt representations. But UniRoute only routes across models — it always sends the full prompt. We extend UniRoute to **jointly route across models AND compression levels**, using TTC's bear API as both the compression engine and the cost-saving mechanism. This is a novel contribution: compression-aware routing with theoretical grounding from UniRoute's framework.

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
        m ∈ user_models
        cost(m, a) ≤ budget.max_cost_per_request
        agg_range.min ≤ a ≤ agg_range.max
```

λ controls the cost-quality tradeoff. Sweep λ from 0 → ∞ to trace a deferral curve.

The key design advantage: the routing table stores per-cluster stats for ALL models and ALL aggressiveness levels. Filtering to any user-defined subset is just a lookup-time operation. No retraining needed when users pick different model pools.

---

## Models

| Model | ID | Provider | Input $/1M | Output $/1M |
|---|---|---|---|---|
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | Anthropic | $1.00 | $5.00 |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | Anthropic | $3.00 | $15.00 |
| GPT-4o-mini | `gpt-4o-mini` | OpenAI | $0.15 | $0.60 |

LLM settings: `temperature=0`, `max_tokens=100`.

## Compression

- **Engine**: TTC bear API (`bear-1.2`)
- **Endpoint**: `POST https://api.thetokencompany.com/v1/compress`
- **Aggressiveness levels**: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
- **API constraints**: aggressiveness must be >0 and <1.0; agg=0.0 is handled by skipping the API call entirely
- **Cost**: $0.05 per 1M tokens removed (negligible)

## Benchmarks

- **SQuAD 2.0**: 150 reading comprehension questions with context passages (seed=42)
- **FinQA**: 150 financial QA questions from SEC 10-K filings (seed=44)
- **Total**: 300 prompts

## Evaluation

**Primary metric: LLM-as-judge** — GPT-4o-mini evaluates each response as "correct" or "incorrect". This replaced F1/EM/CA which were unreliable (EM near zero due to verbose answers, F1 penalized correct-but-verbose responses).

---

## Results (LLM Judge, 3,600 calls)

### Overall Accuracy by Model
| Model | Accuracy | Avg Cost/Request |
|---|---|---|
| Claude Sonnet | 89.0% | $0.00122 |
| Claude Haiku | 84.9% | $0.00046 |
| GPT-4o-mini | 79.7% | $0.00004 |

### Accuracy by Aggressiveness
| Agg | Accuracy |
|---|---|
| 0.0 | 91.2% |
| 0.3 | 89.0% |
| 0.6 | 85.6% |
| 0.9 | 72.3% |

### Router Performance (80/20 train/test split, 60 test prompts)
| Strategy | Accuracy | Cost |
|---|---|---|
| Sonnet, no compression | 98.3% | $0.00123 |
| Haiku, no compression | 96.7% | $0.00044 |
| GPT-4o-mini, no compression | 95.0% | $0.00004 |
| **Router λ=0 (quality)** | **95.0%** | **$0.00058** |
| **Router high λ (cost)** | **88.3%** | **$0.00004** |

### Key Findings
1. **Sonnet is most accurate and most compression-resistant** — but 30x more expensive than GPT-4o-mini
2. **SQuAD2 degrades faster under compression than FinQA** — reading comprehension needs more context intact
3. **Aggressive compression can increase cost** — models produce verbose output on garbled inputs (output tokens cost 5-15x more)
4. **The routing heatmap shows real adaptive behavior** — at low λ, the router picks Claude models with no compression; as λ increases, it transitions to GPT-4o-mini with per-cluster compression levels
5. **Bear compression cost is negligible** (<$0.01) relative to LLM cost ($2+)

---

## Pipeline

### Scripts (run in order)

| Script | Purpose |
|---|---|
| `01_prepare_data.py` | Load SQuAD 2.0 + FinQA from HuggingFace, save to `data/` |
| `02_grid_search.py` | Run all (prompt, agg, model) combos with auto-resume |
| `03_validate.py` | Quick smoke test (2 prompts × all combos) |
| `04_build_router.py` | Embed prompts, K-means clustering, compute cluster stats |
| `05_evaluate.py` | Deferral curves, AUC, QNC vs baselines |
| `06_visualize.py` | Generate all plots |
| `07_llm_judge.py` | Async LLM-as-judge (direct API calls) |
| `07_llm_judge_batch.py` | LLM-as-judge via OpenAI Batch API |

### Core Modules

| Module | Purpose |
|---|---|
| `router/compress.py` | TTC bear API wrapper; skips API at agg=0.0 |
| `router/llm.py` | Unified LLM caller (Anthropic + OpenAI, sync + async) |
| `router/evaluate.py` | EM, F1, CA metrics + cost calculation |
| `router/router.py` | Router class: embed → cluster → score → pick best |

### Grid Search Features
- **Async batching**: 10 concurrent LLM calls via `asyncio.gather`
- **Auto-resume**: Checkpoints every 50 calls, resumes from latest checkpoint
- **Compression cache**: Bear results cached to JSON, never re-called
- **Retry logic**: Phase 3 retries all failed combos with longer delays
- **Failure tracking**: Collects and retries failed compressions and LLM calls

---

## Visualizations

6 plots generated by `06_visualize.py`:

1. **Deferral curve** — Router vs fixed baselines (accuracy vs cost, log scale)
2. **Compression vs accuracy** — All three metrics (judge, F1, CA) by model
3. **Cost vs accuracy scatter** — Every (model, agg) combination
4. **Routing heatmap** — Best (model, agg) per cluster at different λ values
5. **Benchmark comparison** — SQuAD2 vs FinQA accuracy by model and aggressiveness
6. **Cost breakdown** — LLM cost vs bear cost (stacked bar)

---

## What This Enables for TTC

1. **Auto-aggressiveness**: Remove the manual aggressiveness parameter — the router picks it
2. **Model-aware compression**: Different models tolerate different compression levels
3. **Dynamic model pools**: Users pick any subset of models, router adapts at inference time
4. **Budget-constrained optimization**: Stay under $/request ceiling while maximizing accuracy
5. **Adding new models is cheap**: Only need 300 × 10 = 3,000 LLM calls + judge calls, no retraining

## Future Work

- Replace K-means with learned cluster map (UniRoute LearnedMap)
- Tune K (currently 20; try 5, 10, 15, 30)
- Cross-validation instead of single 80/20 split
- Use bear's token-level confidence as additional clustering features
- GRPO-based online learning from production API usage
- Extend to conversation history compression (sliding aggressiveness)
- Per-customer learned λ based on usage patterns
- Streamlit demo app for interactive routing
