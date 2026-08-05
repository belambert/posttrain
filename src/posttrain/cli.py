"""Command line entry points for the post-training stages."""

from typing import Annotated

import typer

from posttrain.config import SftConfig

app = typer.Typer(help="Post-train Qwen3-1.7B-Base.", no_args_is_help=True)

SetOpt = Annotated[
    list[str],
    typer.Option("--set", "-s", help="Override a config key, e.g. -s lr=1e-5."),
]


@app.callback()
def main() -> None:
    """Keeps `sft` a named subcommand; Typer would otherwise collapse a lone
    command to the top level, and later stages add dpo and rl alongside it."""


@app.command()
def sft(config: str = "config/sft.toml", set_: SetOpt = []) -> None:
    """Supervised finetuning on the Tulu 3 SFT mixture."""
    cfg = SftConfig.load(config, set_)
    typer.echo(f"training with {cfg}")

    from posttrain.sft import run_sft  # deferred so --help does not import torch

    run_sft(cfg)


@app.command()
def check(config: str = "config/smoke.toml", set_: SetOpt = []) -> None:
    """Show which tokens the loss mask actually covers, and exit 1 if wrong."""
    from posttrain.check import check_masking

    if not check_masking(SftConfig.load(config, set_)):
        raise typer.Exit(1)
