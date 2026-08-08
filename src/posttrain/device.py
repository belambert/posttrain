"""Device selection, shared by training and sampling."""

import torch


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
