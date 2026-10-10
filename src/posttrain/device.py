"""Device selection, shared by training, sampling, and evaluation."""


def resolve_device() -> str:
    import torch  # deferred so evaluate.py, and its tests, stay torch-free

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
