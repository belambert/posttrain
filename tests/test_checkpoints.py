import json

import pytest

from posttrain.checkpoints import (
    is_resumable,
    last_resumable,
    prune_optimizer_states,
)


def write_safetensors(path, nbytes=16, drop=0):
    """A real safetensors layout: 8-byte header length, JSON header, tensor data."""
    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, nbytes]}}
    ).encode()
    blob = len(header).to_bytes(8, "little") + header + b"\0" * nbytes
    path.write_bytes(blob[: len(blob) - drop] if drop else blob)


def make_full_ckpt(root, step, drop=0):
    """A checkpoint the trainer could resume from, unless `drop` truncates weights."""
    d = root / f"checkpoint-{step}"
    d.mkdir()
    write_safetensors(d / "model.safetensors", drop=drop)
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (d / "scheduler.pt").write_bytes(b"s")
    (d / "optimizer.pt").write_bytes(b"o")
    return d


def make_ckpts(root, steps, size=1000):
    for s in steps:
        d = root / f"checkpoint-{s}"
        d.mkdir()
        (d / "optimizer.pt").write_bytes(b"x" * size)
        (d / "model.safetensors").write_bytes(b"m")
    return root


def surviving(root):
    return sorted(
        int(d.name.split("-")[1])
        for d in root.iterdir()
        if (d / "optimizer.pt").exists()
    )


def test_keeps_the_two_newest(tmp_path):
    make_ckpts(tmp_path, [200, 400, 600, 800])
    freed = prune_optimizer_states(tmp_path, keep=2)

    assert surviving(tmp_path) == [600, 800]
    assert freed == 2000


def test_orders_numerically_not_lexicographically(tmp_path):
    """checkpoint-900 sorts after checkpoint-1000 as a string."""
    make_ckpts(tmp_path, [900, 1000, 1100])
    prune_optimizer_states(tmp_path, keep=2)

    assert surviving(tmp_path) == [1000, 1100]


def test_models_are_never_touched(tmp_path):
    make_ckpts(tmp_path, [100, 200, 300])
    prune_optimizer_states(tmp_path, keep=1)

    assert len(list(tmp_path.glob("checkpoint-*/model.safetensors"))) == 3


def test_is_idempotent(tmp_path):
    make_ckpts(tmp_path, [100, 200, 300])
    prune_optimizer_states(tmp_path, keep=2)
    assert prune_optimizer_states(tmp_path, keep=2) == 0


@pytest.mark.parametrize("keep,expected", [(0, []), (1, [300]), (9, [100, 200, 300])])
def test_keep_counts(tmp_path, keep, expected):
    """keep=0 must strip everything — a naive [:-0] slice would keep them all."""
    make_ckpts(tmp_path, [100, 200, 300])
    prune_optimizer_states(tmp_path, keep=keep)

    assert surviving(tmp_path) == expected


def test_complete_checkpoint_is_resumable(tmp_path):
    assert is_resumable(make_full_ckpt(tmp_path, 100))


def test_truncated_weights_rejected(tmp_path):
    """The file exists and looks fine to a presence check — the bytes are missing."""
    assert not is_resumable(make_full_ckpt(tmp_path, 100, drop=8))


@pytest.mark.parametrize(
    "missing", ["optimizer.pt", "scheduler.pt", "trainer_state.json"]
)
def test_missing_resume_file_rejected(tmp_path, missing):
    ckpt = make_full_ckpt(tmp_path, 100)
    (ckpt / missing).unlink()
    assert not is_resumable(ckpt)


def test_corrupt_trainer_state_rejected(tmp_path):
    ckpt = make_full_ckpt(tmp_path, 100)
    (ckpt / "trainer_state.json").write_text('{"global_step": 10')  # cut off mid-write
    assert not is_resumable(ckpt)


def test_missing_shard_rejected(tmp_path):
    ckpt = make_full_ckpt(tmp_path, 100)
    (ckpt / "model.safetensors").unlink()
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}}
        )
    )
    write_safetensors(ckpt / "shard-1.safetensors")
    assert not is_resumable(ckpt)  # shard-2 never landed


def test_falls_back_past_a_partial_checkpoint(tmp_path):
    make_full_ckpt(tmp_path, 200)
    make_full_ckpt(tmp_path, 400, drop=8)  # killed mid-save, but has the highest step
    assert last_resumable(tmp_path) == str(tmp_path / "checkpoint-200")


def test_none_when_nothing_is_resumable(tmp_path):
    make_full_ckpt(tmp_path, 200, drop=8)
    assert last_resumable(tmp_path) is None


def test_pruned_checkpoint_is_not_resumable(tmp_path):
    """Old checkpoints lose optimizer state by design; they stay loadable, not resumable."""
    make_full_ckpt(tmp_path, 200)
    make_full_ckpt(tmp_path, 400)
    prune_optimizer_states(tmp_path, keep=1)

    assert last_resumable(tmp_path) == str(tmp_path / "checkpoint-400")
