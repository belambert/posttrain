"""Stage one: supervised finetuning of Qwen3-1.7B-Base on the Tulu 3 SFT mixture."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, clone_chat_template

from posttrain.config import SftConfig
from posttrain.data import load_sft


def run_sft(cfg: SftConfig) -> str:
    """Run SFT and return the output dir."""
    trainer = build_trainer(cfg)
    trainer.train()

    trainer.save_model(cfg.out_dir)
    trainer.processing_class.save_pretrained(cfg.out_dir)
    return cfg.out_dir


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

    return SFTTrainer(
        model=model,
        args=_sft_config(cfg, device),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
    )


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
        packing=False,  # packing would blur the assistant mask across examples
        assistant_only_loss=True,
        gradient_checkpointing=True,
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


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
