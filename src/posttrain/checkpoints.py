"""Checkpoint housekeeping.

Optimizer state is ~80% of a checkpoint's size but is only ever read back when
resuming, so keeping it on every checkpoint costs most of the disk for nothing.
Two are kept rather than one: a crash mid-write can leave the newest unusable.
"""

import re
from pathlib import Path

OPTIMIZER_FILES = ("optimizer.pt", "optimizer.bin", "scaler.pt")


def prune_optimizer_states(out_dir: str | Path, keep: int = 2) -> int:
    """Drop optimizer state from all but the `keep` newest checkpoints; bytes freed."""
    ckpts = sorted(Path(out_dir).glob("checkpoint-*"), key=_step)
    stale = ckpts[:-keep] if keep else ckpts

    freed = 0
    for ckpt in stale:
        for name in OPTIMIZER_FILES:
            f = ckpt / name
            if f.is_file():
                freed += f.stat().st_size
                f.unlink()

    return freed


def _step(path: Path) -> int:
    """Sort by step number — lexicographically, checkpoint-900 sorts after 1000."""
    m = re.search(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else -1
