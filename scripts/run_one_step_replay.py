"""Run Phase 3A one-step replay with compressed prefix KV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, eager_attention_forward


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvidcache.capture import load_model_and_tokenizer, load_prompt, normalize_past_key_values, prepare_prompt_text
from kvidcache.codec import encode_anchor_group_residual, encode_previous_token_residual
from kvidcache.metrics import cosine_similarity, mean_squared_energy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model name or path.")
    parser.add_argument("--prompt-file", default="prompts/code_prompt.txt", help="Prompt file to analyze.")
    parser.add_argument("--group-size", type=int, default=16, help="Token group size for the codecs.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, for example `cpu` or `cuda`.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of prompt tokens to keep.")
    return parser


def format_shape(value: torch.Tensor | torch.Size | tuple[int, ...] | list[int]) -> str:
    shape = value.shape if hasattr(value, "shape") else value
    return "x".join(str(dim) for dim in shape)


def tokenize_prompt(tokenizer, prompt: str, max_tokens: int, device: torch.device) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Apply the chat template and tokenize the prompt once."""

    formatted_prompt = prepare_prompt_text(tokenizer, prompt)
    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    return formatted_prompt, encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def compress_prefix_cache(
    prefix_past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    group_size: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Reconstruct prefix KV with the current best codec choice."""

    reconstructed_layers = []
    for raw_k, raw_v in prefix_past_key_values:
        reconstructed_k = encode_previous_token_residual(
            raw_k,
            bit_width=8,
            group_size=group_size,
            codec_name="k-prev-int8",
        ).reconstructed.to(dtype=raw_k.dtype)
        reconstructed_v = encode_anchor_group_residual(
            raw_v,
            bit_width=4,
            group_size=group_size,
            codec_name="v-anchor-int4",
        ).reconstructed.to(dtype=raw_v.dtype)
        reconstructed_layers.append((reconstructed_k, reconstructed_v))
    return tuple(reconstructed_layers)


def capture_raw_one_step_outputs(
    model,
    last_token_id: torch.Tensor,
    full_attention_mask: torch.Tensor,
    prefix_cache_object,
    prefix_length: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Run one raw decode step and capture per-layer attention outputs."""

    attention_outputs: list[torch.Tensor] = []
    hook_handles = []

    def make_hook():
        def hook(module, args, output):
            attention_outputs.append(output[0].detach())

        return hook

    for decoder_layer in model.model.layers:
        hook_handles.append(decoder_layer.self_attn.register_forward_hook(make_hook()))

    try:
        outputs = model(
            input_ids=last_token_id,
            attention_mask=full_attention_mask,
            past_key_values=prefix_cache_object,
            use_cache=True,
            cache_position=torch.tensor([prefix_length], device=last_token_id.device),
            position_ids=torch.tensor([[prefix_length]], device=last_token_id.device),
        )
    finally:
        for handle in hook_handles:
            handle.remove()

    return attention_outputs, outputs.logits.detach()


def run_attention_with_prefix(attention, hidden_states: torch.Tensor, position_embeddings, past_k: torch.Tensor, past_v: torch.Tensor) -> torch.Tensor:
    """Run one attention step for a single token against raw or reconstructed prefix KV."""

    hidden_shape = (*hidden_states.shape[:-1], -1, attention.head_dim)
    query_states = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    full_k = torch.cat([past_k.to(dtype=key_states.dtype), key_states], dim=-2)
    full_v = torch.cat([past_v.to(dtype=value_states.dtype), value_states], dim=-2)
    attn_output, _ = eager_attention_forward(
        attention,
        query_states,
        full_k,
        full_v,
        attention_mask=None,
        scaling=attention.scaling,
        dropout=0.0,
    )
    attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous()
    return attention.o_proj(attn_output)


def replay_one_step_with_reconstructed_prefix(
    model,
    last_token_id: torch.Tensor,
    reconstructed_prefix_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    prefix_length: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Replay one token through the decoder using reconstructed prefix KV."""

    hidden_states = model.model.embed_tokens(last_token_id)
    position_ids = torch.tensor([[prefix_length]], device=last_token_id.device)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    attention_outputs: list[torch.Tensor] = []

    for layer_index, decoder_layer in enumerate(model.model.layers):
        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)

        past_k, past_v = reconstructed_prefix_kv[layer_index]
        attn_output = run_attention_with_prefix(
            decoder_layer.self_attn,
            hidden_states,
            position_embeddings,
            past_k,
            past_v,
        )
        attention_outputs.append(attn_output.detach())
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return attention_outputs, logits.detach()


def topk_overlap(raw_logits: torch.Tensor, reconstructed_logits: torch.Tensor, k: int) -> int:
    """Count overlap between top-k token sets."""

    raw_topk = set(torch.topk(raw_logits[0, -1], k=k).indices.tolist())
    reconstructed_topk = set(torch.topk(reconstructed_logits[0, -1], k=k).indices.tolist())
    return len(raw_topk.intersection(reconstructed_topk))


def print_layer_table(rows: list[dict[str, float | int]]) -> None:
    headers = [
        "layer",
        "attn_cosine",
        "attn_mse",
    ]
    formatted_rows = [
        [
            str(int(row["layer"])),
            f"{float(row['attention_output_cosine']):.8f}",
            f"{float(row['attention_output_mse']):.8f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in formatted_rows))
        for index, header in enumerate(headers)
    ]

    def format_row(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in formatted_rows:
        print(format_row(row))


def main() -> int:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"

    prompt = load_prompt(args.prompt_file)
    _, input_ids, attention_mask = tokenize_prompt(tokenizer, prompt, args.max_tokens, device)
    if input_ids.shape[-1] < 2:
        raise ValueError("Need at least two tokens to build a prefix cache and replay the held-out token.")

    prefix_input_ids = input_ids[:, :-1]
    prefix_attention_mask = attention_mask[:, :-1]
    last_token_id = input_ids[:, -1:]
    full_attention_mask = attention_mask
    prefix_length = int(prefix_input_ids.shape[-1])

    with torch.no_grad():
        prefix_outputs = model(
            input_ids=prefix_input_ids,
            attention_mask=prefix_attention_mask,
            use_cache=True,
        )

    raw_prefix_kv = normalize_past_key_values(prefix_outputs.past_key_values)
    reconstructed_prefix_kv = compress_prefix_cache(raw_prefix_kv, group_size=args.group_size)

    raw_attention_outputs, raw_logits = capture_raw_one_step_outputs(
        model=model,
        last_token_id=last_token_id,
        full_attention_mask=full_attention_mask,
        prefix_cache_object=prefix_outputs.past_key_values,
        prefix_length=prefix_length,
    )
    reconstructed_attention_outputs, reconstructed_logits = replay_one_step_with_reconstructed_prefix(
        model=model,
        last_token_id=last_token_id,
        reconstructed_prefix_kv=reconstructed_prefix_kv,
        prefix_length=prefix_length,
    )

    layer_rows: list[dict[str, float | int]] = []
    for layer_index, (raw_attn, reconstructed_attn) in enumerate(zip(raw_attention_outputs, reconstructed_attention_outputs)):
        layer_rows.append(
            {
                "layer": layer_index,
                "attention_output_cosine": cosine_similarity(raw_attn, reconstructed_attn),
                "attention_output_mse": mean_squared_energy(raw_attn.float() - reconstructed_attn.float()),
            }
        )

    average_attention_cosine = sum(float(row["attention_output_cosine"]) for row in layer_rows) / len(layer_rows)
    worst_layer_attention_cosine = min(float(row["attention_output_cosine"]) for row in layer_rows)
    average_attention_mse = sum(float(row["attention_output_mse"]) for row in layer_rows) / len(layer_rows)

    logit_cosine = cosine_similarity(raw_logits, reconstructed_logits)
    logit_mse = mean_squared_energy(raw_logits.float() - reconstructed_logits.float())
    top1_token_match = int(torch.argmax(raw_logits[0, -1]).item() == torch.argmax(reconstructed_logits[0, -1]).item())
    top5_overlap = topk_overlap(raw_logits, reconstructed_logits, k=5)

    print(f"model name: {args.model}")
    print(f"prompt token count: {input_ids.shape[-1]}")
    print(f"prefix token count: {prefix_length}")
    print(f"held-out replay token id: {int(last_token_id.item())}")
    print(f"K shape: {format_shape(raw_prefix_kv[0][0].shape)}")
    print(f"V shape: {format_shape(raw_prefix_kv[0][1].shape)}")
    print(f"group size: {args.group_size}")
    print()

    print_layer_table(layer_rows)
    print()
    print(f"average attention output cosine: {average_attention_cosine:.8f}")
    print(f"worst-layer attention output cosine: {worst_layer_attention_cosine:.8f}")
    print(f"average attention output MSE: {average_attention_mse:.8f}")
    print(f"logit cosine: {logit_cosine:.8f}")
    print(f"logit MSE: {logit_mse:.8f}")
    print(f"top-1 token match: {top1_token_match}")
    print(f"top-5 overlap: {top5_overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
