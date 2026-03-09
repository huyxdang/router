# Data Collection Specification

## Overview

Every API call in the grid search logs a comprehensive record. This dataset powers the cluster-based router (UniRoute approach), evaluation, and visualization.

---

## Grid Dimensions

```
Prompts:           150 per benchmark × 2 benchmarks = 300 prompts
Benchmarks:        SQuAD 2.0 (reading comprehension) + FinQA (financial QA)
Aggressiveness:    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] = 10 levels
Models:            [claude-haiku, claude-sonnet, gpt-4o-mini] = 3 models

Full grid:
  Bear calls:    300 prompts × 9 agg levels (0.0 skips API) = 2,700
  LLM calls:     300 prompts × 10 agg levels × 3 models = 9,000
  Judge calls:   9,000 (GPT-4o-mini as evaluator)

Completed so far:
  4 agg levels [0.0, 0.3, 0.6, 0.9]: 3,600 LLM calls + 3,600 judge calls ✓
  6 new agg levels [0.1, 0.2, 0.4, 0.5, 0.7, 0.8]: 5,400 LLM calls (in progress)
```

---

## Record Schema

```python
record = {
    # ── Prompt Metadata ──
    "prompt_id": "squad2_042",
    "benchmark": "squad2",                    # "squad2" or "finqa"

    # ── Compression ──
    "aggressiveness": 0.3,
    "original_input_tokens": 842,             # from bear API (0 when agg=0.0)
    "compressed_tokens": 589,                 # from bear API (0 when agg=0.0)
    "compression_ratio": 0.699,               # compressed / original (1.0 when agg=0.0)
    "tokens_removed": 253,
    "removal_rate": 0.301,

    # ── LLM Call ──
    "model_name": "claude-haiku",             # human-readable name
    "model_id": "claude-haiku-4-5-20251001",  # API model string
    "model_provider": "anthropic",            # "anthropic" or "openai"
    "model_cost_per_1m_input": 1.00,
    "model_cost_per_1m_output": 5.00,
    "llm_response": "The capital is Paris.",
    "llm_input_tokens": 589,                  # from LLM API usage
    "llm_output_tokens": 45,                  # from LLM API usage

    # ── Evaluation (automated) ──
    "correct": 1,                             # exact match (case-insensitive)
    "f1_score": 0.92,                         # token-level F1 overlap
    "contains_answer": 1,                     # ground truth substring in response

    # ── Evaluation (LLM judge) ──
    "llm_judge": "correct",                   # "correct" or "incorrect"
    "llm_judge_correct": 1,                   # binary version of above

    # ── Cost ──
    "input_cost_usd": 0.000471,
    "output_cost_usd": 0.000180,
    "total_llm_cost_usd": 0.000651,
    "bear_cost_usd": 0.0000127,               # tokens_removed × $0.05/1M
    "total_cost_usd": 0.000664,

    # ── Metadata ──
    "latency_seconds": 1.34,
    "timestamp": "2026-03-09T14:23:01Z",
    "trial_id": "run_001",
}
```

---

## Storage

| File | Format | Contents |
|---|---|---|
| `results/grid_results.parquet` | Parquet | All grid search records |
| `results/grid_results_judged.parquet` | Parquet | + LLM judge verdicts |
| `results/grid_results_clustered.parquet` | Parquet | + cluster assignments |
| `results/grid_results_checkpoint.parquet` | Parquet | Auto-resume checkpoint |
| `results/compressed_cache.json` | JSON | Bear compression cache (avoid re-calling) |
| `results/router.pkl` | Pickle | K-means model + cluster stats + embeddings |
| `results/evaluation.json` | JSON | Evaluation metrics + deferral curves |
| `results/plots/` | PNG | All generated visualizations |
| `data/squad2_subset.json` | JSON | 150 SQuAD 2.0 prompts |
| `data/finqa_subset.json` | JSON | 150 FinQA prompts |

---

## Actual Costs

```
First run (4 agg levels, 3,600 LLM calls):
  Total LLM cost:   $2.05
  Total bear cost:   $0.01
  Total:             $2.06

Judge cost (3,600 calls to GPT-4o-mini):
  ~$0.05 (tiny tokens per call)

Estimated for full grid (10 agg levels, 9,000 calls):
  ~$5-6 total
```

---

## Auto-Resume System

The grid search (`02_grid_search.py`) supports auto-resume:

- Saves checkpoints every 50 successful LLM calls to `grid_results_checkpoint.parquet`
- On restart, loads whichever file has more records (checkpoint or final results)
- Tracks completed `(prompt_id, aggressiveness, model_name)` tuples
- Skips already-completed combos
- Phase 3 retries all failed combos with longer delays

The compression cache (`compressed_cache.json`) persists bear API results so compressions are never repeated.

---

## Evaluation Metrics

**Primary: LLM-as-judge** — GPT-4o-mini evaluates each response as "correct" or "incorrect". This is the most reliable metric.

Secondary (automated, less reliable):
- **Exact Match (EM)**: Near zero for all models — too strict (models say "The answer is X" instead of "X")
- **F1 Score**: Token overlap — penalizes verbose but correct answers
- **Contains Answer (CA)**: Substring check — false positives on partial matches

---

## Adding a New Model

1. Use `compressed_cache.json` to skip bear API calls
2. Run only new LLM calls: 300 × 10 agg levels = 3,000 calls
3. Run LLM judge on new results
4. Append to grid results and rebuild router
