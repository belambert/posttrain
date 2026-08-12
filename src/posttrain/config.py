"""Training configs: TOML on disk, dataclasses in memory, `--set` overrides on top."""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from types import UnionType
from typing import Iterable, Self, get_args, get_origin, get_type_hints

BASE = "Qwen/Qwen3-1.7B-Base"
INSTRUCT = "Qwen/Qwen3-1.7B"
TULU3_SFT = "allenai/tulu-3-sft-mixture"


@dataclass
class SftConfig:
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
    logging_steps: int = 10
    group_by_length: bool = True  # batch similar lengths so padding stays tight
    gradient_checkpointing: bool = True  # ~30% slower, much cheaper on memory
    resume: bool = True  # pick up from the last checkpoint in out_dir
    device: str | None = None  # cpu, mps, cuda; autodetected when unset
    seed: int = 0
    wandb_project: str | None = None  # unset disables wandb entirely
    run_name: str | None = None

    @classmethod
    def load(cls, path: str | Path, overrides: Iterable[str] = ()) -> Self:
        """Read a TOML config, then apply `key=value` override strings."""
        data = tomllib.loads(Path(path).read_text())
        data.update(_parse_overrides(overrides, cls))

        if unknown := set(data) - {f.name for f in fields(cls)}:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)


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

    if typ is bool:
        return raw.lower() in ("1", "true", "yes")
    return typ(raw)
