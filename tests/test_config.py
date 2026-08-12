import pytest

from posttrain.config import SftConfig

SFT_TOML = "config/sft.toml"
SMOKE_TOML = "config/smoke.toml"
PILOT_TOML = "config/pilot.toml"


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


def test_pilot_stays_comparable_to_the_full_run():
    """A throughput number is only useful if the shape of the step matches."""
    pilot, full = SftConfig.load(PILOT_TOML), SftConfig.load(SFT_TOML)
    assert pilot.n == 50000 and pilot.n < 939000
    for field in ("lr", "batch_size", "grad_accum", "max_length", "base"):
        assert getattr(pilot, field) == getattr(full, field)


def test_smoke_still_takes_a_step():
    """It pins a tiny batch: inheriting the recipe's 128 would give it zero steps."""
    cfg = SftConfig.load(SMOKE_TOML)
    n_train = cfg.n - min(cfg.eval_n, cfg.n // 10)
    assert n_train // (cfg.batch_size * cfg.grad_accum) >= 1


def test_length_grouping_defaults_on_and_is_overridable():
    assert SftConfig.load(SFT_TOML).group_by_length is True
    assert SftConfig.load(SFT_TOML, ["group_by_length=false"]).group_by_length is False


def test_long_run_knobs_default_safe_and_override():
    cfg = SftConfig.load(SFT_TOML)
    assert cfg.resume is True and cfg.gradient_checkpointing is True

    off = SftConfig.load(SFT_TOML, ["resume=false", "gradient_checkpointing=false"])
    assert off.resume is False and off.gradient_checkpointing is False


def test_checkpoints_are_all_kept():
    cfg = SftConfig.load(SFT_TOML)
    assert cfg.save_total_limit is None
    assert SftConfig.load(SFT_TOML, ["save_total_limit=2"]).save_total_limit == 2


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
