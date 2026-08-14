"""Benchmarking checkpoints with lighteval.

Runs the harness as a subprocess rather than through its Python API: lighteval
reworked that API in 0.13 and shells out cleanly, so the CLI is the stable surface.
"""

import json
import subprocess
from pathlib import Path

from posttrain.checkpoints import step_of
from posttrain.config import EvalConfig
from posttrain.device import resolve_device


def run_eval(cfg: EvalConfig) -> tuple[list[str], list[str]]:
    """Benchmark the selected checkpoints; returns (evaluated, failed).

    One model failing does not abort the sweep — across a dozen checkpoints and
    several hours, losing the finished results to a late failure is worse than
    reporting the gap.
    """
    device = cfg.device or resolve_device()
    models = [str(p) for p in select_checkpoints(cfg.checkpoints, cfg.stride)]
    if cfg.baseline:
        models.append(cfg.baseline)

    done, failed = [], []
    for model in models:
        print(f"\n=== evaluating {model}", flush=True)  # subprocess writes unbuffered
        try:
            subprocess.run(_command(model, cfg, device), check=True)
            done.append(model)
        except subprocess.CalledProcessError as e:
            print(f"!!! {model} failed (exit {e.returncode}), continuing", flush=True)
            failed.append(model)

    return done, failed


def select_checkpoints(root: str | Path, stride: int) -> list[Path]:
    """Every `stride`-th checkpoint, always including the newest.

    A directory holding no checkpoint-* subdirs is treated as a checkpoint itself,
    so `--checkpoints runs/sft` works on a finished run.
    """
    root = Path(root)
    ckpts = sorted(root.glob("checkpoint-*"), key=step_of)
    if not ckpts:
        return [root]

    picked = ckpts[::stride] if stride > 0 else ckpts
    if ckpts[-1] not in picked:
        picked.append(ckpts[-1])
    return picked


def collect_results(out_dir: str | Path) -> dict[str, dict[str, float]]:
    """Flatten lighteval's per-run JSON into {model: {task: score}}."""
    table: dict[str, dict[str, float]] = {}
    for f in sorted(Path(out_dir).rglob("results_*.json")):
        data = json.loads(f.read_text())
        model = data.get("config_general", {}).get("model_name") or f.parent.name

        scores = table.setdefault(model, {})
        for task, metrics in data.get("results", {}).items():
            if task == "all":
                continue  # lighteval's own cross-task aggregate, duplicating the rest

            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not name.endswith("_stderr"):
                    scores[f"{task}:{name}"] = float(value)

    return table


def _command(model: str, cfg: EvalConfig, device: str) -> list[str]:
    model_args = (
        f"model_name={model},dtype={_dtype(device)},batch_size={cfg.batch_size}"
    )
    cmd = [
        "lighteval",
        "accelerate",
        model_args,
        ",".join(cfg.tasks),
        "--output-dir",
        cfg.out_dir,
    ]
    if cfg.max_samples:
        cmd += ["--max-samples", str(cfg.max_samples)]
    return cmd


def _dtype(device: str) -> str:
    return "bfloat16" if device == "cuda" else "float32"
