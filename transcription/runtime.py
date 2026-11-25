"""Lazy inference dependencies and GPU lifecycle helpers."""

from __future__ import annotations

import gc
import os
from typing import Any

# Normal transcription is deliberately offline.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"


# Loaded only after argument parsing and local-model validation.
np: Any = None
torch: Any = None
whisper: Any = None


def load_runtime_dependencies() -> None:
    global np, torch, whisper
    import numpy as numpy_module
    import torch as torch_module
    import whisper as whisper_module

    np = numpy_module
    torch = torch_module
    whisper = whisper_module


def clear_gpu() -> None:
    if torch is None:
        return
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def print_gpu_memory(prefix: str = "") -> None:
    if torch is None or not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"{prefix}CUDA: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved")
