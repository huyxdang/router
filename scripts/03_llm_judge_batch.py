"""Alias entrypoint for the LLM judge batch step.

Preferred pipeline command:
  python scripts/03_llm_judge_batch.py [submit|status|download]

Internally forwards to scripts/06_llm_judge_batch.py.
"""

from __future__ import annotations

import os
import runpy


if __name__ == "__main__":
    target = os.path.join(os.path.dirname(__file__), "06_llm_judge_batch.py")
    runpy.run_path(target, run_name="__main__")
