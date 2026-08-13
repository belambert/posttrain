"""Configs: TOML on disk, dataclasses in memory, `--set` overrides on top."""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Iterable, Self, get_args, get_origin, get_type_hints

BASE = "Qwen/Qwen3-1.7B-Base"
INSTRUCT = "Qwen/Qwen3-1.7B"
TULU3_SFT = "allenai/tulu-3-sft-mixture"

# few-shot counts follow the conventional setting for each benchmark
DEFAULT_TASKS = [
    "ifeval|0",
    "gsm8k|5",
    "mmlu|5",
    "bigbench_hard|3",
    "truthfulqa:mc|0",
    "math_500|4",
    "gpqa:diamond|0",
]


class TomlConfig:
    """TOML loading with typed `key=value` overrides, shared by every stage."""

    @classmethod
    def load(cls, path: str | Path, overrides: Iterable[str] = ()) -> Self:
        """Read a TOML config, then apply `key=value` override strings."""
        data = tomllib.loads(Path(path).read_text())
        data.update(_parse_overrides(overrides, cls))

        if unknown := set(data) - {f.name for f in fields(cls)}:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)


@dataclass
class EvalConfig(TomlConfig):
    """Which checkpoints to benchmark, and with what."""

    checkpoints: str = "runs/sft"  # a checkpoint dir, or one containing checkpoint-*
    stride: int = 10  # evaluate every Nth checkpoint; the last is always included
    baseline: str | None = INSTRUCT  # the model this project is measured against
    tasks: list[str] = field(default_factory=lambda: list(DEFAULT_TASKS))
    out_dir: str = "runs/evals"
    batch_size: int = 8
    max_samples: int | None = None  # cap per task, for smoke-testing the harness
    device: str | None = None


@dataclass
class SftConfig(TomlConfig):
    """Hyperparameters for the SFT stage; defaults follow the Tulu 3 recipe."""

    out_dir: str
    base: str = BASE
    template_src: str = INSTRUCT
    dataset: str = TULU3_SFT
    n: int | None = None  # subsample the ~939k-row mixture
    epochs: float = 2.0
    lr: float = 5e-6
    batch_size: int = 2
    grad_accum: int = 64
    max_length: int = 4096
    eval_n: int = 500
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int | None = None  # None keeps every checkpoint
    keep_optimizer_states: int = 2  # only the newest few need to be resumable
    logging_steps: int = 10
    dataset_num_proc: int | None = None  # None uses every core
    group_by_length: bool = True  # batch similar lengths so padding stays tight
    gradient_checkpointing: bool = True  # ~30% slower, much cheaper on memory
    resume: bool = True  # pick up from the last checkpoint in out_dir
    device: str | None = None  # cpu, mps, cuda; autodetected when unset
    seed: int = 0
    wandb_project: str | None = None  # unset disables wandb entirely
    run_name: str | None = None


def _parse_overrides(overrides: Iterable[str], cls: type) -> dict:
    hints = get_type_hints(cls)
    out = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")

        key, raw = item.split("=", 1)
        if key not in hints:
            raise ValueError(f"unknown config key: {key!r}")
        out[key] = _coerce(raw, hints[key])

    return out


def _coerce(raw: str, typ: type):
    """Cast an override string to the field's declared type."""
    if get_origin(typ) is UnionType:  # e.g. int | None
        if raw.lower() in ("none", "null", ""):
            return None
        typ = next(t for t in get_args(typ) if t is not type(None))

    if get_origin(typ) is list:  # e.g. -s tasks=ifeval|0,gsm8k|5
        return [item for item in raw.split(",") if item]
    if typ is bool:
        return raw.lower() in ("1", "true", "yes")
    return typ(raw)
