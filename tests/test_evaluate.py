import json

from posttrain.config import EvalConfig
from posttrain.evaluate import _command, collect_results, select_checkpoints


def make_ckpts(root, steps):
    for s in steps:
        (root / f"checkpoint-{s}").mkdir()
    return root


def names(paths):
    return [p.name for p in paths]


def test_stride_always_includes_the_newest(tmp_path):
    make_ckpts(tmp_path, [200, 400, 600, 800, 1000])
    picked = select_checkpoints(tmp_path, stride=2)

    assert names(picked) == ["checkpoint-200", "checkpoint-600", "checkpoint-1000"]


def test_newest_appended_when_stride_would_skip_it(tmp_path):
    make_ckpts(tmp_path, [200, 400, 600, 800])
    assert "checkpoint-800" in names(select_checkpoints(tmp_path, stride=3))


def test_orders_numerically(tmp_path):
    """checkpoint-900 sorts after checkpoint-1000 as a string."""
    make_ckpts(tmp_path, [900, 1000, 1100])
    assert names(select_checkpoints(tmp_path, stride=1))[-1] == "checkpoint-1100"


def test_plain_dir_is_its_own_checkpoint(tmp_path):
    """A finished run saves the model at the top level, not under checkpoint-*."""
    assert select_checkpoints(tmp_path, stride=10) == [tmp_path]


def test_command_shape():
    cfg = EvalConfig(tasks=["ifeval|0", "gsm8k|5"], batch_size=4, max_samples=8)
    cmd = _command("runs/sft/checkpoint-200", cfg, "cuda")

    assert cmd[:2] == ["lighteval", "accelerate"]
    assert "model_name=runs/sft/checkpoint-200" in cmd[2]
    assert "dtype=bfloat16" in cmd[2] and "batch_size=4" in cmd[2]
    assert "ifeval|0,gsm8k|5" in cmd
    assert "--max-samples" in cmd


def test_command_avoids_bf16_off_cuda():
    assert "dtype=float32" in _command("m", EvalConfig(), "mps")[2]


def test_collect_results_skips_stderr(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "results_2026.json").write_text(
        json.dumps(
            {
                "config_general": {"model_name": "runs/sft/checkpoint-200"},
                "results": {
                    "ifeval|0": {"prompt_acc": 0.42, "prompt_acc_stderr": 0.01}
                },
            }
        )
    )

    table = collect_results(tmp_path)
    assert table == {"runs/sft/checkpoint-200": {"ifeval|0:prompt_acc": 0.42}}
