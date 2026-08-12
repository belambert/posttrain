"""Stage one: supervised finetuning of Qwen3-1.7B-Base on the Tulu 3 SFT mixture."""

import os

import torch
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer, clone_chat_template

from posttrain.config import SftConfig
from posttrain.data import load_sft
from posttrain.device import resolve_device


def run_sft(cfg: SftConfig) -> str:
    """Run SFT and return the output dir."""
    trainer = build_trainer(cfg)

    ckpt = _last_checkpoint(cfg)
    if ckpt:
        print(f"resuming from {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)

    trainer.save_model(cfg.out_dir)
    trainer.processing_class.save_pretrained(cfg.out_dir)
    return cfg.out_dir


def _last_checkpoint(cfg: SftConfig) -> str | None:
    """The full run spans days, so an interruption shouldn't cost the whole thing."""
    if not cfg.resume or not os.path.isdir(cfg.out_dir):
        return None
    return get_last_checkpoint(cfg.out_dir)


def build_trainer(cfg: SftConfig) -> SFTTrainer:
    """Assemble the trainer; shared with `posttrain check` so both see one path."""
    if cfg.wandb_project:
        # the HF integration reads the project from the environment, not from args
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    device = cfg.device or resolve_device()
    model, tok = load_base(cfg.base, device)

    # the base checkpoint has no usable chat format: this installs Qwen3's ChatML
    # template, switches eos to <|im_end|>, and resizes embeddings for the new tokens
    model, tok, _ = clone_chat_template(model, tok, cfg.template_src)

    train_ds, eval_ds = load_sft(cfg.dataset, n=cfg.n, eval_n=cfg.eval_n, seed=cfg.seed)

    return _FixedEvalTrainer(
        model=model,
        args=_sft_config(cfg, device),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
    )


class _FixedEvalTrainer(SFTTrainer):
    """Evaluate in a fixed order so eval_loss stays comparable.

    The base sampler follows `train_sampling_strategy`, so length grouping reshuffles
    the eval set on every call from the global RNG. Since eval loss is averaged per
    batch, that moves the number without the model changing.
    """

    def _get_eval_sampler(self, eval_dataset) -> SequentialSampler:
        return SequentialSampler(eval_dataset)


def _sft_config(cfg: SftConfig, device: str) -> SFTConfig:
    return SFTConfig(
        output_dir=cfg.out_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        max_length=cfg.max_length,
        # packing is unavailable: its default bfd strategy forces padding-free, which
        # needs FlashAttention 2/3 kernels that have no aarch64 build. Grouping by
        # length is the substitute — it can't remove padding, only shrink it.
        packing=False,
        train_sampling_strategy="group_by_length" if cfg.group_by_length else "random",
        assistant_only_loss=True,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=device == "cuda",
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        seed=cfg.seed,
        report_to=["wandb"] if cfg.wandb_project else [],
        run_name=cfg.run_name,
    )


def load_base(name: str, device: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model on `device`; bf16 only where it is well supported."""
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    return model.to(device), AutoTokenizer.from_pretrained(name)
