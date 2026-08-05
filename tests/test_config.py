import pytest

from posttrain.config import SftConfig

SFT_TOML = "config/sft.toml"
SMOKE_TOML = "config/smoke.toml"


def test_loads_shipped_recipe():
    cfg = SftConfig.load(SFT_TOML)
    assert cfg.base == "Qwen/Qwen3-1.7B-Base"
    assert cfg.batch_size * cfg.grad_accum == 128  # Tulu 3 effective batch


def test_overrides_are_typed():
    cfg = SftConfig.load(SFT_TOML, ["lr=1e-5", "n=1000", "out_dir=runs/x"])
    assert cfg.lr == 1e-5 and isinstance(cfg.lr, float)
    assert cfg.n == 1000 and isinstance(cfg.n, int)
    assert cfg.out_dir == "runs/x"


def test_override_clears_optional():
    assert SftConfig.load(SFT_TOML, ["n=none"]).n is None


def test_wandb_is_opt_in():
    assert SftConfig.load(SFT_TOML).wandb_project == "posttrain"
    assert SftConfig.load(SMOKE_TOML).wandb_project is None
    assert SftConfig.load(SFT_TOML, ["wandb_project=none"]).wandb_project is None


@pytest.mark.parametrize("bad", ["lr", "nope=1"])
def test_bad_overrides_fail_fast(bad):
    with pytest.raises(ValueError):
        SftConfig.load(SFT_TOML, [bad])


def test_unknown_toml_key_fails_fast(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('out_dir = "runs/x"\nlearnign_rate = 1e-5\n')
    with pytest.raises(ValueError, match="unknown config keys"):
        SftConfig.load(p)
