"""LLM judge via Qwen2.5-7B-Instruct on Modal (vLLM).

Provides:
  - Modal app + Judge class (GPU inference)
  - judge_responses(): callable from any script, handles Modal context
"""

from __future__ import annotations

import modal

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_DIR = "/models"

JUDGE_TEMPLATE = (
    "You are an evaluation judge. Determine if the response correctly answers "
    "the question.\n\n"
    "Ground truth answer: {ground_truth}\n\n"
    "Model response: {response}\n\n"
    "Is the model's response correct? It doesn't need to match exactly — "
    "it just needs to contain or convey the same answer as the ground truth.\n\n"
    'Reply with ONLY "correct" or "incorrect".'
)

app = modal.App("bear-router-judge")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.7.3", "transformers>=4.48.2,<5.0.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

model_volume = modal.Volume.from_name(
    "bear-judge-model-cache", create_if_missing=True
)


@app.cls(
    image=vllm_image,
    gpu="A10G",
    volumes={MODEL_DIR: model_volume},
    timeout=900,
    scaledown_window=120,
)
class Judge:
    @modal.enter()
    def load_model(self):
        from pathlib import Path

        from huggingface_hub import snapshot_download
        from vllm import LLM, SamplingParams

        model_path = Path(MODEL_DIR) / MODEL_ID
        if not model_path.exists():
            print(f"Downloading {MODEL_ID}...")
            snapshot_download(MODEL_ID, local_dir=str(model_path))
            model_volume.commit()

        self.llm = LLM(model=str(model_path), max_model_len=2048, dtype="half")
        self.params = SamplingParams(temperature=0, max_tokens=10)

    @modal.method()
    def judge_batch(self, prompts: list[str]) -> list[str]:
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = self.llm.chat(conversations, self.params)
        return [o.outputs[0].text.strip() for o in outputs]


def parse_verdict(text: str) -> str:
    """Parse model output into 'correct' or 'incorrect'."""
    t = " ".join(text.strip().lower().split())
    if t == "correct":
        return "correct"
    if t == "incorrect":
        return "incorrect"
    if "incorrect" in t or "not correct" in t:
        return "incorrect"
    if "correct" in t:
        return "correct"
    return "incorrect"


def judge_responses(
    ground_truths: list[str],
    responses: list[str],
    batch_size: int = 500,
) -> list[str]:
    """Judge a list of (ground_truth, response) pairs via Modal GPU workers.

    Returns list of 'correct' or 'incorrect' strings, one per input pair.
    Handles Modal context automatically — works from any Python script.
    """
    if not ground_truths:
        return []

    prompts = [
        JUDGE_TEMPLATE.format(ground_truth=gt, response=resp)
        for gt, resp in zip(ground_truths, responses)
    ]

    batches = [
        prompts[i : i + batch_size]
        for i in range(0, len(prompts), batch_size)
    ]

    def _execute():
        judge = Judge()
        all_verdicts = []
        for i, batch_verdicts in enumerate(judge.judge_batch.map(batches)):
            parsed = [parse_verdict(v) for v in batch_verdicts]
            all_verdicts.extend(parsed)
            print(
                f"  Judge: batch {i + 1}/{len(batches)} | "
                f"{len(all_verdicts)}/{len(prompts)} done"
            )
        return all_verdicts

    with app.run():
        return _execute()
