"""Run Phase 2B offline sweeps over K and V codec variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvidcache.capture import capture_prompt_kv, load_model_and_tokenizer, load_prompt
from kvidcache.codec import (
    encode_anchor_group_residual,
    encode_previous_token_residual,
    encode_raw_tensor,
    qk_logit_mae,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model name or path.")
    parser.add_argument("--prompt-file", default="prompts/code_prompt.txt", help="Prompt file to analyze.")
    parser.add_argument(
        "--group-size",
        type=int,
        default=16,
        choices=[4, 8, 16, 32, 64, 128],
        help="Token block size for predictors and quant scales.",
    )
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
    """Compute the last-token query for one decoder layer using captured inputs."""

    decoder_layer = model.model.layers[layer_index]
    attention = decoder_layer.self_attn
    hidden_shape = (*layer_input.shape[:-1], -1, attention.head_dim)

    query_states = attention.q_proj(layer_input).view(hidden_shape).transpose(1, 2)
    key_states = attention.k_proj(layer_input).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, _ = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return query_states[..., -1:, :]


def build_k_codecs(group_size: int):
    return [
        ("raw", 16, lambda tensor: encode_raw_tensor(tensor, 16, group_size, "k-raw-int16")),
        ("raw", 8, lambda tensor: encode_raw_tensor(tensor, 8, group_size, "k-raw-int8")),
        ("raw", 6, lambda tensor: encode_raw_tensor(tensor, 6, group_size, "k-raw-int6")),
        ("raw", 4, lambda tensor: encode_raw_tensor(tensor, 4, group_size, "k-raw-int4")),
        ("prev-residual", 16, lambda tensor: encode_previous_token_residual(tensor, 16, group_size, "k-prev-int16")),
        ("prev-residual", 8, lambda tensor: encode_previous_token_residual(tensor, 8, group_size, "k-prev-int8")),
        ("prev-residual", 6, lambda tensor: encode_previous_token_residual(tensor, 6, group_size, "k-prev-int6")),
        ("prev-residual", 4, lambda tensor: encode_previous_token_residual(tensor, 4, group_size, "k-prev-int4")),
    ]


def build_v_codecs(group_size: int):
    return [
        ("raw", 16, lambda tensor: encode_raw_tensor(tensor, 16, group_size, "v-raw-int16")),
        ("raw", 8, lambda tensor: encode_raw_tensor(tensor, 8, group_size, "v-raw-int8")),
        ("raw", 6, lambda tensor: encode_raw_tensor(tensor, 6, group_size, "v-raw-int6")),
        ("raw", 4, lambda tensor: encode_raw_tensor(tensor, 4, group_size, "v-raw-int4")),
        ("anchor-residual", 16, lambda tensor: encode_anchor_group_residual(tensor, 16, group_size, "v-anchor-int16")),
        ("anchor-residual", 8, lambda tensor: encode_anchor_group_residual(tensor, 8, group_size, "v-anchor-int8")),
        ("anchor-residual", 6, lambda tensor: encode_anchor_group_residual(tensor, 6, group_size, "v-anchor-int6")),
        ("anchor-residual", 4, lambda tensor: encode_anchor_group_residual(tensor, 4, group_size, "v-anchor-int4")),
    ]


def print_summary_table(rows: list[dict[str, float | int | str]]) -> None:
    headers = [
        "K codec",
        "V codec",
        "K bits",
        "V bits",
        "K cosine",
        "V cosine",
        "qK logit MAE",
        "K ratio",
        "V ratio",
        "full KV ratio",
    ]
    formatted_rows = [
        [
            str(row["k_codec_name"]),
            str(row["v_codec_name"]),
            str(int(row["k_bits"])),
            str(int(row["v_bits"])),
            f"{float(row['k_cosine']):.6f}",
            f"{float(row['v_cosine']):.6f}",
            f"{float(row['qk_logit_mae']):.6f}",
            f"{float(row['k_compression_ratio']):.4f}",
            f"{float(row['v_compression_ratio']):.4f}",
            f"{float(row['full_kv_compression_ratio']):.4f}",
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


def save_json_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


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

    k_codec_specs = build_k_codecs(args.group_size)
    v_codec_specs = build_v_codecs(args.group_size)

    qk_queries = []
    raw_k_repeated = []
    num_kv_groups = []
    attn_scalings = []
    for layer_index, (layer_input, position_embeddings) in enumerate(
        zip(capture.layer_attention_inputs, capture.layer_position_embeddings)
    ):
        if position_embeddings is None:
            raise ValueError(f"Missing position embeddings for layer {layer_index}.")

        attention = model.model.layers[layer_index].self_attn
        qk_queries.append(compute_last_token_query(model, layer_index, layer_input, position_embeddings))
        raw_k_repeated.append(repeat_kv(capture.past_key_values[layer_index][0], attention.num_key_value_groups))
        num_kv_groups.append(attention.num_key_value_groups)
        attn_scalings.append(attention.scaling)

    cached_k_results: dict[tuple[str, int], list[tuple[object, float]]] = {}
    for k_name, k_bits, k_encoder in k_codec_specs:
        per_layer = []
        for layer_index, (raw_k, last_query, repeated_raw_k, scaling, num_groups) in enumerate(
            zip(
                [layer_cache[0] for layer_cache in capture.past_key_values],
                qk_queries,
                raw_k_repeated,
                attn_scalings,
                num_kv_groups,
            )
        ):
            codec_result = k_encoder(raw_k)
            reconstructed_k_for_logits = repeat_kv(codec_result.reconstructed, num_groups)
            logit_mae = qk_logit_mae(last_query, repeated_raw_k, reconstructed_k_for_logits, scaling)
            per_layer.append((codec_result, logit_mae))
        cached_k_results[(k_name, k_bits)] = per_layer

    cached_v_results: dict[tuple[str, int], list[object]] = {}
    for v_name, v_bits, v_encoder in v_codec_specs:
        cached_v_results[(v_name, v_bits)] = [v_encoder(raw_v) for _, raw_v in capture.past_key_values]

    aggregate_rows: list[dict[str, float | int | str]] = []
    per_layer_rows: list[dict[str, float | int | str]] = []

    for k_name, k_bits, _ in k_codec_specs:
        k_results = cached_k_results[(k_name, k_bits)]
        total_raw_k_bytes = sum(result.raw_bytes for result, _ in k_results)
        total_codec_k_bytes = sum(result.codec_bytes for result, _ in k_results)
        average_k_cosine = sum(result.cosine_similarity for result, _ in k_results) / len(k_results)
        average_qk_logit_mae = sum(logit_mae for _, logit_mae in k_results) / len(k_results)

        for v_name, v_bits, _ in v_codec_specs:
            v_results = cached_v_results[(v_name, v_bits)]
            total_raw_v_bytes = sum(result.raw_bytes for result in v_results)
            total_codec_v_bytes = sum(result.codec_bytes for result in v_results)
            average_v_cosine = sum(result.cosine_similarity for result in v_results) / len(v_results)

            aggregate_rows.append(
                {
                    "k_codec_name": k_name,
                    "v_codec_name": v_name,
                    "k_bits": k_bits,
                    "v_bits": v_bits,
                    "k_cosine": average_k_cosine,
                    "v_cosine": average_v_cosine,
                    "qk_logit_mae": average_qk_logit_mae,
                    "k_compression_ratio": total_raw_k_bytes / total_codec_k_bytes,
                    "v_compression_ratio": total_raw_v_bytes / total_codec_v_bytes,
                    "full_kv_compression_ratio": (total_raw_k_bytes + total_raw_v_bytes)
                    / (total_codec_k_bytes + total_codec_v_bytes),
                }
            )

            for layer_index, ((k_result, logit_mae), v_result) in enumerate(zip(k_results, v_results)):
                per_layer_rows.append(
                    {
                        "layer": layer_index,
                        "k_codec_name": k_name,
                        "v_codec_name": v_name,
                        "k_bits": k_bits,
                        "v_bits": v_bits,
                        "k_mse": k_result.mse,
                        "k_cosine": k_result.cosine_similarity,
                        "v_mse": v_result.mse,
                        "v_cosine": v_result.cosine_similarity,
                        "qk_logit_mae": logit_mae,
                        "raw_k_bytes": k_result.raw_bytes,
                        "codec_k_bytes": k_result.codec_bytes,
                        "raw_v_bytes": v_result.raw_bytes,
                        "codec_v_bytes": v_result.codec_bytes,
                    }
                )

    print(f"model name: {capture.model_name}")
    print(f"prompt token count: {capture.input_ids.shape[-1]}")
    print(f"number of layers: {len(capture.past_key_values)}")
    print(f"K shape: {format_shape(capture.past_key_values[0][0].shape)}")
    print(f"V shape: {format_shape(capture.past_key_values[0][1].shape)}")
    print(f"group size: {args.group_size}")
    print()

    print_summary_table(aggregate_rows)

    report = {
        "model_name": capture.model_name,
        "prompt_token_count": int(capture.input_ids.shape[-1]),
        "group_size": args.group_size,
        "device": str(device),
        "max_tokens": args.max_tokens,
        "aggregate_metrics": aggregate_rows,
        "per_layer_metrics": per_layer_rows,
    }
    save_json_report(report, REPO_ROOT / "outputs" / "kv_codec_sweep.json")
    print()
    print("Saved JSON report to outputs/kv_codec_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
