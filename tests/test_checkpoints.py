import pytest

from posttrain.checkpoints import prune_optimizer_states


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
