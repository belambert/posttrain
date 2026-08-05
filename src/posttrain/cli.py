"""Command line entry points for the post-training stages."""

from typing import Annotated

import typer

from posttrain.config import SftConfig

app = typer.Typer(help="Post-train Qwen3-1.7B-Base.", no_args_is_help=True)

SetOpt = Annotated[
    list[str],
    typer.Option("--set", "-s", help="Override a config key, e.g. -s lr=1e-5."),
]


@app.command()
def sft(config: str = "config/sft.toml", set_: SetOpt = []) -> None:
    """Supervised finetuning on the Tulu 3 SFT mixture."""
    cfg = SftConfig.load(config, set_)
    typer.echo(f"training with {cfg}")

    from posttrain.sft import run_sft  # deferred so --help does not import torch

    run_sft(cfg)
