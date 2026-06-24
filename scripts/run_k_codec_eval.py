"""Run Phase 2A K-only predictive residual codec evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvidcache.capture import capture_prompt_kv, load_model_and_tokenizer, load_prompt
from kvidcache.codec import encode_previous_token_residual, qk_logit_mae


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model name or path.")
    parser.add_argument("--prompt-file", default="prompts/code_prompt.txt", help="Prompt file to analyze.")
    parser.add_argument("--group-size", type=int, default=16, help="Residual block size for int8 scales.")
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


def compute_last_token_query(model, layer_index: int, layer_input: torch.Tensor, position_embeddings) -> torch.Tensor:
    """Compute the last-token query for one decoder layer using the captured forward inputs."""

    decoder_layer = model.model.layers[layer_index]
    attention = decoder_layer.self_attn
    hidden_shape = (*layer_input.shape[:-1], -1, attention.head_dim)

    query_states = attention.q_proj(layer_input).view(hidden_shape).transpose(1, 2)
    key_states = attention.k_proj(layer_input).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, _ = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return query_states[..., -1:, :]


def print_layer_table(rows: list[dict[str, float | int]]) -> None:
    headers = [
        "layer",
        "k_mse",
        "k_cosine",
        "qk_logit_mae",
        "raw_k_bytes",
        "codec_k_bytes",
        "k_ratio",
    ]
    formatted_rows = [
        [
            str(int(row["layer"])),
            f"{float(row['k_reconstruction_mse']):.8f}",
            f"{float(row['k_cosine_similarity']):.8f}",
            f"{float(row['qk_logit_mae']):.8f}",
            str(int(row["raw_k_bytes"])),
            str(int(row["codec_k_bytes"])),
            f"{float(row['k_compression_ratio']):.4f}",
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
    prompt = load_prompt(args.prompt_file)

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    capture = capture_prompt_kv(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        model_name=args.model,
        max_tokens=args.max_tokens,
        capture_attention_inputs=True,
    )

    if capture.layer_attention_inputs is None or capture.layer_position_embeddings is None:
        raise ValueError("Layer attention inputs were not captured.")

    total_raw_k_bytes = 0
    total_codec_k_bytes = 0
    total_raw_v_bytes = 0
    layer_rows: list[dict[str, float | int]] = []

    for layer_index, ((raw_k, raw_v), layer_input, position_embeddings) in enumerate(
        zip(capture.past_key_values, capture.layer_attention_inputs, capture.layer_position_embeddings)
    ):
        if position_embeddings is None:
            raise ValueError(f"Missing position embeddings for layer {layer_index}.")

        codec_result = encode_previous_token_residual(
            raw_k,
            bit_width=8,
            group_size=args.group_size,
            codec_name="k-prev-int8",
        )
        last_query = compute_last_token_query(model, layer_index, layer_input, position_embeddings)
        raw_k_for_logits = repeat_kv(raw_k, model.model.layers[layer_index].self_attn.num_key_value_groups)
        reconstructed_k_for_logits = repeat_kv(
            codec_result.reconstructed, model.model.layers[layer_index].self_attn.num_key_value_groups
        )
        logit_mae = qk_logit_mae(
            last_query=last_query,
            raw_k=raw_k_for_logits,
            reconstructed_k=reconstructed_k_for_logits,
            scaling=model.model.layers[layer_index].self_attn.scaling,
        )

        total_raw_k_bytes += codec_result.raw_bytes
        total_codec_k_bytes += codec_result.codec_bytes
        total_raw_v_bytes += raw_v.numel() * 2

        layer_rows.append(
            {
                "layer": layer_index,
                "k_reconstruction_mse": codec_result.mse,
                "k_cosine_similarity": codec_result.cosine_similarity,
                "qk_logit_mae": logit_mae,
                "raw_k_bytes": codec_result.raw_bytes,
                "codec_k_bytes": codec_result.codec_bytes,
                "k_compression_ratio": codec_result.raw_bytes / codec_result.codec_bytes,
            }
        )

    average_k_mse = sum(float(row["k_reconstruction_mse"]) for row in layer_rows) / len(layer_rows)
    average_k_cosine = sum(float(row["k_cosine_similarity"]) for row in layer_rows) / len(layer_rows)
    average_qk_logit_mae = sum(float(row["qk_logit_mae"]) for row in layer_rows) / len(layer_rows)
    k_only_compression_ratio = total_raw_k_bytes / total_codec_k_bytes
    raw_full_kv_bytes = total_raw_k_bytes + total_raw_v_bytes
    codec_full_kv_bytes = total_codec_k_bytes + total_raw_v_bytes
    full_kv_compression_ratio = raw_full_kv_bytes / codec_full_kv_bytes

    print(f"model name: {capture.model_name}")
    print(f"prompt token count: {capture.input_ids.shape[-1]}")
    print(f"number of layers: {len(capture.past_key_values)}")
    print(f"K shape: {format_shape(capture.past_key_values[0][0].shape)}")
    print(f"V shape: {format_shape(capture.past_key_values[0][1].shape)}")
    print(f"group size: {args.group_size}")
    print()

    print_layer_table(layer_rows)
    print()
    print(f"average K reconstruction MSE: {average_k_mse:.8f}")
    print(f"average K cosine similarity: {average_k_cosine:.8f}")
    print(f"average q·K logit MAE: {average_qk_logit_mae:.8f}")
    print(f"raw K bytes: {total_raw_k_bytes}")
    print(f"codec K bytes: {total_codec_k_bytes}")
    print(f"estimated compression ratio for K only: {k_only_compression_ratio:.4f}")
    print(f"raw full KV bytes: {raw_full_kv_bytes}")
    print(f"codec full KV bytes with raw V: {codec_full_kv_bytes}")
    print(f"estimated full KV compression with raw V: {full_kv_compression_ratio:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
