"""Verifying the loss mask before committing hours of compute to it.

The failure this exists to catch is silent: ask for assistant masks against a
template with no `{% generation %}` markers and you get an all-zero mask rather
than an error, so training runs to completion having learned nothing.
"""

import typer

from posttrain.config import SftConfig

IGNORE = -100  # HF's "do not compute loss here" label

RED = typer.colors.RED
GREEN = typer.colors.GREEN


def check_masking(cfg: SftConfig, n_examples: int = 3) -> bool:
    """Decode what the first batch actually trains on. True if it looks right."""
    from posttrain.sft import build_trainer  # deferred: pulls in torch

    trainer = build_trainer(cfg)
    tok = trainer.processing_class
    batch = next(iter(trainer.get_train_dataloader()))

    if "labels" not in batch:
        return _banner(False, "the collator produced no labels at all")

    ok = True
    for i in range(min(n_examples, len(batch["labels"]))):
        ids = batch["input_ids"][i].tolist()
        labels = batch["labels"][i].tolist()

        trained = tok.decode([t for t, l in zip(ids, labels) if l != IGNORE])
        n_trained = sum(l != IGNORE for l in labels)
        n_real = sum(l != IGNORE or t != tok.pad_token_id for t, l in zip(ids, labels))

        typer.secho(f"\nexample {i}", bold=True)
        typer.echo(f"  {n_trained} of {n_real} tokens carry loss")
        typer.echo(f"  trained text: {trained[:400]!r}")
        ok &= _judge(trained, n_trained, tok)

    return _banner(
        ok, "loss covers assistant turns only" if ok else "do not train on this"
    )


def _judge(trained: str, n_trained: int, tok) -> bool:
    """Report every way the mask is wrong, rather than stopping at the first."""
    problems = []
    if n_trained == 0:
        problems.append("nothing is trained on — the generation markers are missing")
    if "<|im_start|>user" in trained:
        problems.append("user turns leaked into the loss")
    if n_trained and tok.eos_token not in trained:
        problems.append(f"{tok.eos_token} is masked out — it won't learn to stop")

    for p in problems:
        typer.secho(f"  ✗ {p}", fg=RED, bold=True)
    if not problems:
        typer.secho("  ✓ assistant turns only, stop token included", fg=GREEN)

    return not problems


def _banner(ok: bool, detail: str) -> bool:
    """Make the verdict impossible to miss in a wall of training output."""
    label, color = ("PASS", GREEN) if ok else ("FAIL", RED)
    typer.echo()
    typer.secho(f"  {label}  {detail}  ", fg=typer.colors.BLACK, bg=color, bold=True)
    typer.echo()
    return ok
