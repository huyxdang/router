import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ============ API Keys ============
TTC_API_KEY = os.environ.get("TTC_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# ============ Paths ============
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============ Bear Compression ============
BEAR_API_URL = "https://api.thetokencompany.com/v1/compress"
BEAR_MODEL = "bear-1.2"
BEAR_COST_PER_1M_TOKENS_REMOVED = 0.05  # USD

# ============ Models ============
MODELS = [
    {
        "name": "gpt-4.1-nano",
        "id": "gpt-4.1-nano",
        "provider": "openai",
        "cost_per_1m_input": 0.10,
        "cost_per_1m_output": 0.40,
    },
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
        "name": "gpt-5.4",
        "id": "gpt-5.4",
        "provider": "openai",
        "cost_per_1m_input": 2.50,
        "cost_per_1m_output": 15.00,
    },
]

# ============ OpenRouter ============
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Mapping from our model IDs to OpenRouter model IDs
OPENROUTER_MODEL_IDS = {
    "gpt-4.1-nano": "openai/gpt-4.1-nano",
    "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4.5",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "gpt-5.4": "openai/gpt-5.4",
}

# ============ LLM Settings ============
LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0

SYSTEM_PROMPTS = {
    "squad2": (
        "You are a reading comprehension assistant. Answer the question based only on the provided context. "
        "Be concise and give the answer directly. "
        "If the question cannot be answered from the context, respond with exactly: unanswerable"
    ),
    "finqa": (
        "You are a financial analysis assistant. Answer the question based on the provided SEC filing context. "
        "Be precise with numbers, percentages, and financial terms. Give the answer directly."
    ),
    "coqa": (
        "You are a conversational assistant. Answer the question based on the provided context and conversation history. "
        "Be concise and give the answer directly."
    ),
    "financebench": (
        "You are a financial analysis assistant specializing in SEC filings. Answer the question based on the provided "
        "financial statements and disclosures. Be precise with numbers, percentages, and financial terms. Give the answer directly."
    ),
}
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based on the provided context. "
    "Be concise and give the answer directly."
)

# ============ Compression Tiers ============
AGGRESSIVENESS_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]

# ============ Benchmarks & Data ============
# Training benchmarks: used for clustering, model profiling, grid search
BENCHMARKS = ["squad2", "finqa", "coqa"]
PROMPTS_PER_BENCHMARK = 1000

# Eval-only benchmarks: used for final evaluation, not training
EVAL_BENCHMARKS = ["financebench"]

# Data splits (train / val / test)
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.10
TEST_FRACTION = 0.20
RANDOM_SEED = 42

# ============ Clustering ============
EMBEDDING_MODEL = "text-embedding-3-small" # by OpenAI
K_VALUES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# ============ Grid Search ============
BATCH_SIZE = 10  # concurrent LLM calls
CHECKPOINT_EVERY = 50

# ============ LLM Judge ============
JUDGE_MODEL = "gpt-4o-mini"

# ============ Helpers ============
def get_model_by_name(name: str) -> dict:
    for m in MODELS:
        if m["name"] == name:
            return m
    raise ValueError(f"Unknown model: {name}")
