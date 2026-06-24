"""Model loading and KV cache capture helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class CaptureResult:
    """Container for one prompt capture."""

    model_name: str
    prompt: str
    formatted_prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]


def load_prompt(prompt_file: str | Path) -> str:
    """Read a prompt file and normalize whitespace a little."""

    text = Path(prompt_file).read_text(encoding="utf-8")
    return text.strip()


def load_model_and_tokenizer(model_name: str, device: torch.device) -> tuple[Any, Any]:
    """Load a causal LM and tokenizer on the requested device."""

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        # Decoder-only models often do not define a pad token. Reuse EOS so
        # batched tokenization still works cleanly for this research script.
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prepare_prompt_text(tokenizer: Any, prompt: str) -> str:
    """Apply a chat template when the tokenizer provides one."""

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Some tokenizers expose a template but still fail on template
            # rendering. Falling back to the raw prompt keeps the experiment
            # runnable instead of brittle.
            return prompt

    return prompt


def _normalize_past_key_values(past_key_values: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Convert cache objects into a plain tuple of (key, value) pairs."""

    if past_key_values is None:
        raise ValueError("Model did not return past_key_values.")

    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()

    normalized = tuple(past_key_values) if not isinstance(past_key_values, tuple) else past_key_values
    validated_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

    for layer_index, layer_cache in enumerate(normalized):
        if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) < 2:
            raise ValueError(
                f"Layer {layer_index} cache is not a (key, value) pair. "
                f"Observed type={type(layer_cache)!r}."
            )

        key_tensor, value_tensor = layer_cache[0], layer_cache[1]
        if not isinstance(key_tensor, torch.Tensor) or not isinstance(value_tensor, torch.Tensor):
            raise ValueError(
                f"Layer {layer_index} cache did not contain tensors. "
                f"Observed key type={type(key_tensor)!r}, value type={type(value_tensor)!r}."
            )

        validated_layers.append((key_tensor, value_tensor))

    return tuple(validated_layers)


@torch.no_grad()
def capture_prompt_kv(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    model_name: str,
    max_tokens: int,
) -> CaptureResult:
    """Run the prompt through the model and return the resulting KV cache."""

    formatted_prompt = prepare_prompt_text(tokenizer, prompt)
    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

    return CaptureResult(
        model_name=model_name,
        prompt=prompt,
        formatted_prompt=formatted_prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=_normalize_past_key_values(outputs.past_key_values),
    )
