"""Resolve the CTranslate2 model used by local inference and evaluation."""

import os
from pathlib import Path


PROJECT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "whisper-persian-ct2-int8"


def default_ct2_model_dir() -> str:
    """Prefer an explicit deployment path, otherwise use the project-local model."""
    return os.environ.get("CT2_MODEL_DIR", str(PROJECT_MODEL_DIR))
