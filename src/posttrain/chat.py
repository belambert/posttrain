"""Sampling from a checkpoint — whether the loss bought a usable chat model.

Falling loss and a model that answers well are different claims, and only the
second one is what the comparison against Qwen's own post-train is about.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from posttrain.device import resolve_device


def reply(
    model,
    tok,
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate one assistant turn for a conversation."""
    # enable_thinking=False matches training: the recipe's assistant turns are
    # prefixed with an empty <think> block, so priming keeps inference consistent
    enc = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature or None,
        top_p=0.95,
        pad_token_id=tok.pad_token_id,
    )
    prompt_len = enc["input_ids"].shape[1]
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def load_for_chat(path: str, device: str | None = None):
    """Load a checkpoint and its tokenizer, ready to generate."""
    device = device or resolve_device()
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    return model.to(device).eval(), AutoTokenizer.from_pretrained(path)
