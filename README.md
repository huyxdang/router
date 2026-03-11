# Compression-Aware LLM Router

A compression-aware LLM routing system built for The Token Company. Extends the [UniRoute K-means framework (Google, 2025)](https://arxiv.org/pdf/2502.08773) to jointly route across LLM models and compression tiers. Given a prompt, the router embeds it, finds its nearest cluster, and selects the optimal (model, compression) pair that minimizes `error + lambda * cost`.

Uses TTC's **bear-1.2** compression API to reduce input tokens before routing to one of 4 LLMs across 5 compression tiers, producing **20 virtual models**.

---

## Architecture

```
                          +------------------+
                          |   Input Prompt   |
                          +--------+---------+
                                   |
                          +--------v---------+
                          |  Embed (OpenAI   |
                          |  text-embedding  |
                          |  -3-small, 1536d)|
                          +--------+---------+
                                   |
                          +--------v---------+
                          | Nearest Cluster  |
                          |   (K-means)      |
                          +--------+---------+
                                   |
                          +--------v---------+
                          | Score 20 virtual |
                          | models per       |
                          | cluster stats    |-----> User constraints:
                          | error + l * cost |       - model pool
                          +--------+---------+       - budget ceiling
                                   |                 - compression bounds
                          +--------v---------+       - lambda tradeoff
                          | Best (model,     |
                          |  compression)    |
                          +---+----------+---+
                              |          |
                    +---------v--+  +----v-----------+
                    | Compress   |  | Route to LLM   |
                    | via bear   |  | (GPT / Claude) |
                    +------------+  +----+-----------+
                                         |
                                  +------v------+
                                  |   Output    |
                                  +-------------+
```

## Models

| Model | Provider | Input / Output (per 1M tokens) | Role |
|---|---|---|---|
| gpt-4.1-nano | OpenAI | $0.10 / $0.40 | Ultra-cheap |
| claude-haiku | Anthropic | $1.00 / $5.00 | Mid-tier |
| gpt-5.4 | OpenAI | $2.50 / $15.00 | Premium |
| claude-sonnet | Anthropic | $3.00 / $15.00 | Quality ceiling |

**Compression tiers:** 0%, 20%, 40%, 60%, 80% via bear-1.2 ($0.05 per 1M tokens removed).

## Benchmarks

| Benchmark | Source | Samples | Role |
|---|---|---|---|
| SQuAD 2.0 | HuggingFace | 1,000 | Training (reading comprehension) |
| financial-qa-10K | HuggingFace | 1,000 | Training (financial QA) |
| CoQA | HuggingFace | 1,000 | Training (conversational QA) |
| FinanceBench | Patronus AI | 150 | Evaluation only |

**Data split (training benchmarks):** 70% train (2,100 -- clustering only), 10% val + 20% test (900 -- grid search and evaluation).

**Baselines:** GPT-5.4 Only, OpenRouter (same 4 models), UniRoute No Compression.

## Quick Start

```bash
# Create and activate isolated env (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API keys
cp .env.example .env
# Edit .env: TTC_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY

# Run the pipeline (in order)
python scripts/00_check_apis.py               # Verify all API endpoints
python scripts/01_prepare_data.py             # Download datasets from HuggingFace
python scripts/01_validate.py                 # Smoke test (1 prompt per benchmark)
python scripts/02_grid_search.py              # Async compression + LLM calls (val+test)
python scripts/03_llm_judge_batch.py submit   # Submit judge jobs via OpenAI Batch API
python scripts/03_llm_judge_batch.py status   # Check batch status
python scripts/03_llm_judge_batch.py download # Download judge verdicts
# Or automate all chunks end-to-end:
python scripts/03_llm_judge_batch.py run 800 30  # chunk=800, poll every 30s
python scripts/03_build_router.py             # Embed, cluster, build routing table
python scripts/08_tune_and_cv.py              # K-tuning via cross-validation
python scripts/04_evaluate.py                 # Deferral curves, AUC, QNC, FinanceBench
python scripts/05_visualize.py                # Generate plots
```

## Pipeline Overview

| Step | Script | Description |
|---|---|---|
| 0 | `00_check_apis.py` | Verify TTC, OpenAI, Anthropic, and OpenRouter endpoints are reachable |
| 1a | `01_prepare_data.py` | Download SQuAD 2.0, financial-qa-10K, CoQA, and FinanceBench; format as JSON |
| 1b | `01_validate.py` | Smoke test: 1 prompt per benchmark through compress + LLM |
| 2 | `02_grid_search.py` | Run all (prompt, compression, model) combos on val+test split; auto-resume with checkpoints |
| 3 | `03_llm_judge_batch.py` | Judge answers via OpenAI Batch API (gpt-4o-mini); submit / status / download subcommands |
| 4 | `03_build_router.py` | Embed prompts with text-embedding-3-small, K-means clustering, compute cluster stats |
| 5 | `08_tune_and_cv.py` | Tune K (number of clusters) via cross-validation |
| 6 | `04_evaluate.py` | Deferral curves, AUC, QNC metrics; FinanceBench held-out evaluation |
| 7 | `05_visualize.py` | Generate all plots (heatmaps, deferral curves, cost-quality tradeoffs) |

## File Structure

```
router/
├── config.py                           # API keys, model configs, compression tiers
├── requirements.txt
├── .env                                # API keys (gitignored)
├── .env.example
│
├── router/                             # Core modules
│   ├── compress.py                     # Bear API wrapper (sync + async)
│   ├── llm.py                          # LLM caller: Anthropic + OpenAI + OpenRouter (sync + async)
│   ├── evaluate.py                     # Cost computation + judge prompt
│   ├── router.py                       # Router class (embed -> cluster -> route)
│   ├── data.py                         # Shared data loading utilities
│   ├── embeddings.py                   # OpenAI embeddings with caching
│   ├── clustering.py                   # Cluster stats computation
│   └── scoring.py                      # Scoring formula, deferral curves, AUC, QNC
│
├── scripts/                            # Pipeline scripts (run in order)
│   ├── 00_check_apis.py                # Verify all API endpoints
│   ├── 01_prepare_data.py              # Download and format benchmark data
│   ├── 01_validate.py                  # Smoke test
│   ├── 02_grid_search.py              # Grid search with auto-resume + rate limiting
│   ├── 03_build_router.py             # Embed, K-means, cluster stats
│   ├── 03_llm_judge_batch.py          # Preferred alias for LLM-as-judge batch step
│   ├── 04_evaluate.py                 # Deferral curves, AUC, QNC, FinanceBench
│   ├── 05_visualize.py                # Generate all plots
│   ├── 06_llm_judge_batch.py          # LLM-as-judge via OpenAI Batch API
│   └── 08_tune_and_cv.py             # K-tuning via cross-validation
│
├── data/                               # Benchmark datasets
│   ├── squad2_subset.json
│   ├── finqa_subset.json
│   ├── coqa_subset.json
│   └── financebench_subset.json
│
└── results/                            # Output artifacts
    ├── grid_results_checkpoint.parquet # Grid search checkpoint
    ├── grid_results_judged.parquet     # With LLM judge verdicts
    ├── compressed_cache.json           # Bear compression cache
    ├── router_config.json              # Routing table (portable)
    ├── centroids.npy                   # K-means centroids
    ├── cluster_stats.parquet           # Per-cluster model performance
    └── plots/                          # Generated visualizations
```

## Router Artifacts

The trained router is stored as three portable files (no pickle):

- **`router_config.json`** -- routing table mapping clusters to optimal (model, compression) pairs
- **`centroids.npy`** -- K-means cluster centroids (NumPy array)
- **`cluster_stats.parquet`** -- per-cluster performance statistics for all 20 virtual models

## Requirements

```
anthropic, openai, httpx, pandas, numpy, scikit-learn, matplotlib, datasets, python-dotenv
```

See `requirements.txt` for pinned versions.

## License

See [LICENSE](LICENSE).
