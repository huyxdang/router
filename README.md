# Adaptive Compression Router

**UniRoute-inspired joint model + compression routing using TTC's bear API.**

Extends [UniRoute (Google, 2025)](https://arxiv.org/pdf/2502.08773) to jointly route across LLM models AND compression aggressiveness levels. Given a prompt, the router embeds it, finds its nearest cluster, and selects the optimal (model, aggressiveness) pair that minimizes `error + λ * cost` — adapting to the user's model pool, budget, and quality requirements.

---

## Architecture

```
Prompt → Embed (MiniLM) → Nearest Cluster → Score (model, agg) pairs → Best option → Compress (bear) → LLM → Output
                                                    ↑
                                          User constraints:
                                          - model pool
                                          - budget ceiling
                                          - compression bounds
                                          - λ (cost-quality tradeoff)
```

## Key Results (LLM-as-Judge, 60 held-out prompts)

| Strategy | Accuracy | Avg Cost/Request |
|---|---|---|
| Sonnet, no compression (ceiling) | 98.3% | $0.00123 |
| Haiku, no compression | 96.7% | $0.00044 |
| GPT-4o-mini, no compression | 95.0% | $0.00004 |
| **Router, λ=0 (quality mode)** | **95.0%** | **$0.00058** |
| **Router, high λ (cost mode)** | **88.3%** | **$0.00004** |

The router at quality mode achieves near-Sonnet accuracy at half the cost. At cost mode, it beats fixed GPT-4o-mini by ~5% accuracy at the same price through adaptive per-cluster compression.

## Key Findings

- **Sonnet is most accurate** (95.3%) and most compression-resistant, but 30x more expensive than GPT-4o-mini
- **SQuAD2 degrades faster under compression than FinQA** — the router learns to protect reading comprehension and compress financial questions more aggressively
- **Aggressive compression can increase cost** — models produce more verbose (expensive) output on garbled inputs
- **Bear compression cost is negligible** (<$0.01 total) relative to LLM cost ($2+)
- **The routing heatmap shows real adaptive behavior** — at low λ, router picks Claude with no compression; at high λ, it transitions to GPT-4o-mini with per-cluster compression levels

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
cp .env.example .env
# Edit .env with your TTC_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY

# Run the pipeline
python scripts/01_prepare_data.py      # Download & format benchmark data
python scripts/02_grid_search.py       # Run all (prompt, agg, model) combos
python scripts/07_llm_judge_batch.py submit   # Judge results via OpenAI Batch API
python scripts/07_llm_judge_batch.py download # Download judge verdicts
python scripts/04_build_router.py      # Embed, cluster, build routing table
python scripts/05_evaluate.py          # Evaluate router vs baselines
python scripts/06_visualize.py         # Generate all plots
```

## File Structure

```
router/
├── config.py                          # API keys, model configs, aggressiveness levels
├── requirements.txt
├── .env                               # API keys (gitignored)
├── MODEL.md                           # Detailed project plan & architecture
├── DATA.md                            # Data collection specification
├── CONTEXT.md                         # Context for future development sessions
│
├── router/                            # Core modules
│   ├── compress.py                    # TTC bear API wrapper
│   ├── llm.py                         # LLM caller (Anthropic + OpenAI, sync + async)
│   ├── evaluate.py                    # EM, F1, CA metrics + cost calculation
│   └── router.py                      # Router class (embed → cluster → route)
│
├── scripts/                           # Pipeline scripts (run in order)
│   ├── 01_prepare_data.py             # Load SQuAD 2.0 + FinQA from HuggingFace
│   ├── 02_grid_search.py             # Grid search with auto-resume + retry
│   ├── 03_validate.py                # Quick smoke test (2 prompts × all combos)
│   ├── 04_build_router.py            # Embed prompts, K-means, compute cluster stats
│   ├── 05_evaluate.py                # Deferral curves, AUC, QNC vs baselines
│   ├── 06_visualize.py               # Generate all plots
│   ├── 07_llm_judge.py               # Async LLM-as-judge (direct API)
│   └── 07_llm_judge_batch.py         # LLM-as-judge via OpenAI Batch API
│
├── data/                              # Benchmark datasets (gitignored)
│   ├── squad2_subset.json             # 150 SQuAD 2.0 questions
│   └── finqa_subset.json             # 150 FinQA questions
│
└── results/                           # Output artifacts (gitignored)
    ├── grid_results.parquet           # All grid search records
    ├── grid_results_judged.parquet    # With LLM judge verdicts
    ├── grid_results_clustered.parquet # With cluster assignments
    ├── compressed_cache.json          # Bear compression cache
    ├── router.pkl                     # Trained router (kmeans + cluster stats)
    ├── evaluation.json                # Evaluation metrics
    └── plots/                         # Generated visualizations
```

## Models

| Model | Provider | Cost (input/output per 1M) | Role |
|---|---|---|---|
| Claude Haiku 4.5 | Anthropic | $1.00 / $5.00 | Mid-tier |
| Claude Sonnet 4.6 | Anthropic | $3.00 / $15.00 | Quality ceiling |
| GPT-4o-mini | OpenAI | $0.15 / $0.60 | Cost-efficient |
| GPT-5.4 | OpenAI | $2.50 / $15.00 | Premium |
| GPT-4.1-nano | OpenAI | $0.10 / $0.40 | Ultra-cheap |
| GPT-4.1 | OpenAI | $2.00 / $8.00 | High-quality |

## Benchmarks

- **SQuAD 2.0**: 150 reading comprehension questions (seed=42)
- **FinQA**: 150 financial QA questions from SEC 10-K filings (seed=44)
