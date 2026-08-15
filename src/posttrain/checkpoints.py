"""Checkpoint housekeeping.

Optimizer state is ~80% of a checkpoint's size but is only ever read back when
resuming, so keeping it on every checkpoint costs most of the disk for nothing.
Two are kept rather than one: a crash mid-write can leave the newest unusable.
"""

import json
import re
from pathlib import Path

OPTIMIZER_FILES = ("optimizer.pt", "optimizer.bin", "scaler.pt")
RESUME_FILES = ("trainer_state.json", "scheduler.pt", "optimizer.pt")


def last_resumable(out_dir: str | Path) -> str | None:
    """Newest checkpoint that is actually complete, or None.

    A run killed mid-save — spot preemption, power cut — leaves a checkpoint
    directory whose files are truncated or missing. It has the highest step
    number, so picking the newest blindly resumes from the broken one.
    """
    for ckpt in sorted(Path(out_dir).glob("checkpoint-*"), key=step_of, reverse=True):
        if is_resumable(ckpt):
            return str(ckpt)
        print(f"skipping incomplete checkpoint {ckpt}")

    return None


def is_resumable(ckpt: Path) -> bool:
    """Whether a checkpoint has everything the trainer needs to continue."""
    if not all((ckpt / f).is_file() for f in RESUME_FILES):
        return False
    if not _json_parses(ckpt / "trainer_state.json"):
        return False
    return _weights_intact(ckpt)


def _weights_intact(ckpt: Path) -> bool:
    """Follow the shard index when there is one, then verify each shard."""
    index = ckpt / "model.safetensors.index.json"
    if not index.is_file():
        return _safetensors_intact(ckpt / "model.safetensors")

    try:
        shards = set(json.loads(index.read_text())["weight_map"].values())
    except (OSError, ValueError, KeyError):
        return False

    return all(_safetensors_intact(ckpt / s) for s in shards)


def _safetensors_intact(f: Path) -> bool:
    """Check the declared size against the real one; a truncated file still exists.

    safetensors is 8 bytes of header length, that much JSON, then tensor data at
    the offsets the header declares — so the header says how long the file must be.
    """
    try:
        with f.open("rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            if not 0 < n < 100_000_000:
                return False
            header = json.loads(fh.read(n))
    except (OSError, ValueError):
        return False

    ends = [v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"]
    return f.stat().st_size >= 8 + n + max(ends, default=0)


def _json_parses(f: Path) -> bool:
    try:
        json.loads(f.read_text())
        return True
    except (OSError, ValueError):
        return False


def prune_optimizer_states(out_dir: str | Path, keep: int = 2) -> int:
    """Drop optimizer state from all but the `keep` newest checkpoints; bytes freed."""
    ckpts = sorted(Path(out_dir).glob("checkpoint-*"), key=step_of)
    stale = ckpts[:-keep] if keep else ckpts

    freed = 0
    for ckpt in stale:
        for name in OPTIMIZER_FILES:
            f = ckpt / name
            if f.is_file():
                freed += f.stat().st_size
                f.unlink()

    return freed


def step_of(path: Path) -> int:
    """Sort by step number — lexicographically, checkpoint-900 sorts after 1000."""
    m = re.search(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else -1
