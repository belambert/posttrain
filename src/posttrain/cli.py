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
def chat(
    model: str = "runs/sft",
    prompt: str | None = typer.Option(None, "--prompt", "-p", help="One-shot ask."),
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: str | None = None,
) -> None:
    """Sample from a checkpoint. Without --prompt, opens a multi-turn session."""
    from posttrain.chat import load_for_chat, reply

    m, tok = load_for_chat(model, device)
    kw = dict(max_new_tokens=max_new_tokens, temperature=temperature)

    if prompt:
        typer.echo(reply(m, tok, [{"role": "user", "content": prompt}], **kw))
        return

    typer.secho(f"{model} — blank line or ctrl-d to exit", fg=typer.colors.CYAN)
    messages: list[dict] = []
    while user := typer.prompt("\nyou", default="", show_default=False).strip():
        messages.append({"role": "user", "content": user})
        answer = reply(m, tok, messages, **kw)
        messages.append({"role": "assistant", "content": answer})
        typer.secho(f"\n{answer}", fg=typer.colors.GREEN)


@app.command()
def check(config: str = "config/smoke.toml", set_: SetOpt = []) -> None:
    """Show which tokens the loss mask actually covers, and exit 1 if wrong."""
    from posttrain.check import check_masking

    if not check_masking(SftConfig.load(config, set_)):
        raise typer.Exit(1)
