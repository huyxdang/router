import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ============ API Keys ============
TTC_API_KEY = os.environ.get("TTC_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ============ Bear Compression ============
BEAR_API_URL = "https://api.thetokencompany.com/v1/compress"
BEAR_MODEL = "bear-1.2"
BEAR_COST_PER_1M_TOKENS_REMOVED = 0.05  # USD

# ============ Models ============
MODELS = [
    {
        "name": "claude-haiku",
        "id": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "cost_per_1m_input": 1.00,
        "cost_per_1m_output": 5.00,
    },
    {
        "name": "claude-sonnet",
        "id": "claude-sonnet-4-6",
        "provider": "anthropic",
        "cost_per_1m_input": 3.00,
        "cost_per_1m_output": 15.00,
    },
    {
        "name": "gpt-4o-mini",
        "id": "gpt-4o-mini",
        "provider": "openai",
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60,
    },
]

# ============ Grid Search ============
AGGRESSIVENESS_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PROMPTS_PER_BENCHMARK = 150

# ============ Clustering ============
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_CLUSTERS = 5

# ============ Helpers ============
def get_model_by_name(name: str) -> dict:
    for m in MODELS:
        if m["name"] == name:
            return m
    raise ValueError(f"Unknown model: {name}")
