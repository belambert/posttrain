# posttrain

Taking `Qwen/Qwen3-1.7B-Base` through a full post-training pipeline on a DGX Spark, and
comparing the result against Qwen's own post-trained `Qwen/Qwen3-1.7B`. That comparison is
the point: it measures how close an open recipe gets to Qwen's own.

Stage one — supervised finetuning on the Tulu 3 SFT mixture. Preference
tuning and RL come later.

## Setup

    uv sync

That is all that is needed on either machine. The Spark's GB10 is Blackwell (sm_121) and
needs a CUDA 13 build of PyTorch, so `pyproject.toml` routes torch to the `cu130` index on
Linux and leaves macOS on the stock PyPI wheel for its MPS build. `uv sync` picks the right
one from the platform.

## Running SFT

Three configs, in the order they are meant to be used. First prove the pipeline runs end to
end on 64 examples:

    uv run posttrain sft --config config/smoke.toml

Then a 50k-example pilot, which is the one that tells you what the full run will cost —
divide its throughput into 939k examples times two epochs before committing to it:

    uv run posttrain sft --config config/pilot.toml

Then the real run:

    uv run posttrain sft

Any config key can be overridden from the command line, which keeps one-off experiments out
of version control:

    uv run posttrain sft -s lr=1e-5 -s n=50000

## Verifying before a long run

    uv run posttrain check

This builds the real trainer, pulls the first batch, and decodes exactly which tokens carry
loss — printing the trained text and exiting non-zero if nothing is trained on, if user
turns leak into the loss, or if the stop token is masked out. Worth running before any
multi-hour job, since a broken mask does not raise on its own.

## Tracking

Runs log to Weights & Biases when `wandb_project` is set, which `config/sft.toml` does and
`config/smoke.toml` deliberately does not. Log in once per machine:

    uv run wandb login

To train without it, clear the key:

    uv run posttrain sft -s wandb_project=none

`posttrain check` never logs, whatever the config says — a pre-flight check shouldn't leave
runs behind.

## Configs

Recipes live in `config/` as TOML and are read into `SftConfig` (`src/posttrain/config.py`),
which is the single source of truth for the schema — unknown keys in either the file or a
`--set` override are an error rather than a silent no-op.

| Config              | Purpose                                                            |
| ------------------- | ------------------------------------------------------------------ |
| `config/smoke.toml` | 64 examples, short sequences — a correctness check, not a run.      |
| `config/pilot.toml` | 50k examples, 1 epoch — a real loss curve and a throughput number.  |
| `config/sft.toml`   | The Tulu 3 recipe: 2 epochs, lr 5e-6, effective batch 128.          |

`batch_size * grad_accum` is the effective batch size; keep the product at 128 when trading
one against the other to fit memory.

## Chat template

The base checkpoint ships no usable chat format, so `clone_chat_template` copies Qwen3's
ChatML template across, switches the EOS token from `<|endoftext|>` to `<|im_end|>`, and
resizes the embeddings for the added tokens.

Loss is computed on assistant turns only (`assistant_only_loss=True`). This matters more
than it looks: Qwen3's published template has no `{% generation %}` markers, and asking for
assistant masks against it does not raise — it silently returns an all-zero mask, which
would train on no tokens at all. TRL patches the template itself when the markers are
missing, and warns if any example ends up with zero assistant tokens.

## Development

    uv run pytest
    uv run black . && uv run isort .

The tests cover config loading and CLI wiring only, and deliberately run without torch or
trl installed — `posttrain sft --help` defers the training imports so it stays fast.
