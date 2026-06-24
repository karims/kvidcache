"""Run a Phase 1 residual analysis over model KV cache tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvidcache.capture import capture_prompt_kv, load_model_and_tokenizer, load_prompt
from kvidcache.metrics import summarize_prediction
from kvidcache.predictors import PREDICTORS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--prompt-file",
        default="prompts/code_prompt.txt",
        help="Path to a text file containing the prompt to analyze.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=16,
        help="Anchor spacing used by the anchor predictor.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, for example `cpu` or `cuda`.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of prompt tokens to keep before the forward pass.",
    )
    return parser


def format_shape(value: torch.Tensor | torch.Size | tuple[int, ...] | list[int]) -> str:
    """Format either a tensor or a shape-like object as `d0xd1x...`."""

    shape = value.shape if hasattr(value, "shape") else value
    return "x".join(str(dim) for dim in shape)


def get_model_parameter_device(model: torch.nn.Module) -> torch.device:
    """Report the device of the first model parameter."""

    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Model does not appear to have any parameters.") from exc


def get_peak_cuda_memory_allocated(device: torch.device) -> int:
    """Return peak allocated CUDA memory in bytes for the selected device."""

    if device.type != "cuda":
        return 0

    return int(torch.cuda.max_memory_allocated(device))


def validate_kv_tensor(tensor: torch.Tensor, tensor_name: str, layer_index: int) -> None:
    """Require the standard Hugging Face KV shape used by this experiment."""

    # We expect [batch, kv_heads, seq_len, head_dim]. The token axis is dim 2,
    # which is the axis the predictors operate on in this Phase 1 experiment.
    if tensor.ndim != 4:
        raise ValueError(
            f"Expected {tensor_name} tensor in layer {layer_index} to have shape "
            f"[batch, kv_heads, seq_len, head_dim], but observed {tuple(tensor.shape)}."
        )


def make_layer_record(layer_index: int, tensor_name: str, metric_report) -> dict[str, float | int | str]:
    return {
        "layer": layer_index,
        "tensor": tensor_name,
        "raw_energy": metric_report.raw_energy,
        "residual_energy": metric_report.residual_energy,
        "residual_energy_ratio": metric_report.residual_energy_ratio,
        "cosine_similarity": metric_report.cosine_similarity,
        "mean_absolute_error": metric_report.mean_absolute_error,
    }


def aggregate_metric_records(metric_records: list[dict[str, float | int | str]]) -> dict[str, float]:
    ratios = [float(record["residual_energy_ratio"]) for record in metric_records]
    cosines = [float(record["cosine_similarity"]) for record in metric_records]
    maes = [float(record["mean_absolute_error"]) for record in metric_records]

    return {
        "average_residual_energy_ratio": sum(ratios) / len(ratios),
        "best_layer_residual_energy_ratio": min(ratios),
        "worst_layer_residual_energy_ratio": max(ratios),
        "average_cosine_similarity": sum(cosines) / len(cosines),
        "average_mae": sum(maes) / len(maes),
    }


def analyze_predictor(
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    predictor_name: str,
    predictor_fn,
    group_size: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | str]]]:
    """Return per-layer metrics and per-(predictor, tensor) aggregate rows."""

    per_layer_metrics: list[dict[str, float | int | str]] = []
    aggregate_rows: list[dict[str, float | str]] = []

    for tensor_name, tensor_index in (("K", 0), ("V", 1)):
        tensor_records: list[dict[str, float | int | str]] = []

        for layer_index, layer_cache in enumerate(past_key_values):
            tensor = layer_cache[tensor_index]
            validate_kv_tensor(tensor, tensor_name, layer_index)
            predicted = predictor_fn(tensor, group_size)
            metric_report = summarize_prediction(tensor, predicted)
            record = make_layer_record(layer_index, tensor_name, metric_report)
            tensor_records.append(record)
            per_layer_metrics.append({"predictor": predictor_name, **record})

        aggregate_rows.append(
            {
                "predictor": predictor_name,
                "tensor": tensor_name,
                **aggregate_metric_records(tensor_records),
            }
        )

    return per_layer_metrics, aggregate_rows


def print_summary_table(rows: list[dict[str, float | str]]) -> None:
    headers = [
        "predictor",
        "K/V",
        "avg_ratio",
        "best_layer",
        "worst_layer",
        "avg_cosine",
        "avg_mae",
    ]
    formatted_rows = [
        [
            str(row["predictor"]),
            str(row["tensor"]),
            f"{float(row['average_residual_energy_ratio']):.6f}",
            f"{float(row['best_layer_residual_energy_ratio']):.6f}",
            f"{float(row['worst_layer_residual_energy_ratio']):.6f}",
            f"{float(row['average_cosine_similarity']):.6f}",
            f"{float(row['average_mae']):.6f}",
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
    parser = build_arg_parser()
    args = parser.parse_args()

    device = torch.device(args.device)
    prompt = load_prompt(args.prompt_file)
    cuda_available = torch.cuda.is_available()

    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"selected device: {device}")
    if cuda_available:
        print(f"GPU name: {torch.cuda.get_device_name(device)}")
    else:
        print("GPU name: unavailable")

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    print(f"model parameter device after loading: {get_model_parameter_device(model)}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    capture = capture_prompt_kv(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        model_name=args.model,
        max_tokens=args.max_tokens,
    )
    peak_cuda_memory_allocated = get_peak_cuda_memory_allocated(device)

    if not capture.past_key_values:
        raise ValueError("Model returned an empty past_key_values sequence.")

    key_shape = capture.past_key_values[0][0].shape
    value_shape = capture.past_key_values[0][1].shape
    validate_kv_tensor(capture.past_key_values[0][0], "K", 0)
    validate_kv_tensor(capture.past_key_values[0][1], "V", 0)

    print(f"model name: {capture.model_name}")
    print(f"prompt token count: {capture.input_ids.shape[-1]}")
    print(f"number of layers: {len(capture.past_key_values)}")
    print(f"KV shape: key={format_shape(key_shape)}, value={format_shape(value_shape)}")
    print(f"group size: {args.group_size}")
    print(f"peak CUDA memory allocated after forward pass: {peak_cuda_memory_allocated}")
    print()

    all_per_layer_metrics: list[dict[str, float | int | str]] = []
    all_aggregate_rows: list[dict[str, float | str]] = []

    for spec in PREDICTORS:
        per_layer_metrics, aggregate_rows = analyze_predictor(
            past_key_values=capture.past_key_values,
            predictor_name=spec.name,
            predictor_fn=spec.predictor,
            group_size=args.group_size,
        )
        all_per_layer_metrics.extend(per_layer_metrics)
        all_aggregate_rows.extend(aggregate_rows)

    print_summary_table(all_aggregate_rows)

    report = {
        "model_name": capture.model_name,
        "prompt_token_count": int(capture.input_ids.shape[-1]),
        "group_size": args.group_size,
        "device": str(device),
        "max_tokens": args.max_tokens,
        "per_layer_metrics": all_per_layer_metrics,
        "aggregate_metrics": all_aggregate_rows,
    }
    save_json_report(report, REPO_ROOT / "outputs" / "residual_report.json")
    print()
    print("Saved JSON report to outputs/residual_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
