import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ============ API Keys ============
TTC_API_KEY = os.environ.get("TTC_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

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
        "name": "mistral-large",
        "id": "mistral-large-2512",
        "provider": "mistral",
        "cost_per_1m_input": 0.50,
        "cost_per_1m_output": 1.50,
    },
]

# ============ Grid Search ============
AGGRESSIVENESS_LEVELS = [0.1, 0.4, 0.7, 1.0]
PROMPTS_PER_BENCHMARK = 150

# ============ Clustering ============
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_CLUSTERS = 20

# ============ Helpers ============
def get_model_by_name(name: str) -> dict:
    for m in MODELS:
        if m["name"] == name:
            return m
    raise ValueError(f"Unknown model: {name}")
