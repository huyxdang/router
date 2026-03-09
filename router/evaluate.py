import re
import string
from collections import Counter

from config import BEAR_COST_PER_1M_TOKENS_REMOVED


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truth: str) -> int:
    """Case-insensitive exact match after normalization."""
    return int(_normalize(prediction) == _normalize(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 overlap between prediction and ground truth."""
    pred_tokens = _normalize(prediction).split()
    truth_tokens = _normalize(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_answer(prediction: str, ground_truth: str) -> int:
    """Check if ground truth appears as substring in prediction."""
    return int(_normalize(ground_truth) in _normalize(prediction))


def compute_cost(model_config: dict, input_tokens: int, output_tokens: int,
                 tokens_removed: int = 0) -> dict:
    """Compute USD cost breakdown for a single request."""
    input_cost = input_tokens * model_config["cost_per_1m_input"] / 1_000_000
    output_cost = output_tokens * model_config["cost_per_1m_output"] / 1_000_000
    bear_cost = tokens_removed * BEAR_COST_PER_1M_TOKENS_REMOVED / 1_000_000

    return {
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_llm_cost_usd": input_cost + output_cost,
        "bear_cost_usd": bear_cost,
        "total_cost_usd": input_cost + output_cost + bear_cost,
    }
