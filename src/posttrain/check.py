"""Verifying the loss mask before committing hours of compute to it.

The failure this exists to catch is silent: ask for assistant masks against a
template with no `{% generation %}` markers and you get an all-zero mask rather
than an error, so training runs to completion having learned nothing.
"""

from posttrain.config import SftConfig

IGNORE = -100  # HF's "do not compute loss here" label


def check_masking(cfg: SftConfig, n_examples: int = 3) -> bool:
    """Decode what the first batch actually trains on. True if it looks right."""
    from posttrain.sft import build_trainer  # deferred: pulls in torch

    trainer = build_trainer(cfg)
    tok = trainer.processing_class
    batch = next(iter(trainer.get_train_dataloader()))

    if "labels" not in batch:
        print("FAIL: collator produced no labels")
        return False

    ok = True
    for i in range(min(n_examples, len(batch["labels"]))):
        ids = batch["input_ids"][i].tolist()
        labels = batch["labels"][i].tolist()

        trained = tok.decode([t for t, l in zip(ids, labels) if l != IGNORE])
        n_trained = sum(l != IGNORE for l in labels)
        n_real = sum(l != IGNORE or t != tok.pad_token_id for t, l in zip(ids, labels))

        print(f"\n--- example {i}: {n_trained}/{n_real} tokens trained on")
        print(f"    trained text: {trained[:400]!r}")
        ok &= _judge(trained, n_trained, tok)

    return ok


def _judge(trained: str, n_trained: int, tok) -> bool:
    """Report the ways a mask goes wrong, rather than stopping at the first."""
    ok = True
    if n_trained == 0:
        print("    FAIL: nothing is trained on — the generation markers are missing")
        ok = False

    if "<|im_start|>user" in trained:
        print("    FAIL: user turns are inside the loss")
        ok = False

    if n_trained and tok.eos_token not in trained:
        print(
            f"    FAIL: {tok.eos_token} is masked out — the model won't learn to stop"
        )
        ok = False

    if ok:
        print("    ok: assistant turns only, stop token included")
    return ok
