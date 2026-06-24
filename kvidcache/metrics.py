"""Metrics for comparing raw KV tensors against predictor outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorMetricReport:
    """Summary metrics for one raw tensor and one predicted tensor."""

    raw_energy: float
    residual_energy: float
    residual_energy_ratio: float
    cosine_similarity: float
    mean_absolute_error: float


def mean_squared_energy(tensor: torch.Tensor) -> float:
    """Compute a stable energy proxy as mean squared magnitude."""

    return float(tensor.float().pow(2).mean().item())


def cosine_similarity(raw_tensor: torch.Tensor, predicted_tensor: torch.Tensor) -> float:
    """Compute cosine similarity over the flattened tensor."""

    raw_flat = raw_tensor.float().reshape(-1)
    predicted_flat = predicted_tensor.float().reshape(-1)
    raw_norm = torch.linalg.vector_norm(raw_flat)
    predicted_norm = torch.linalg.vector_norm(predicted_flat)
    denom = raw_norm * predicted_norm

    if float(denom.item()) == 0.0:
        return 0.0

    return float((torch.dot(raw_flat, predicted_flat) / denom).item())


def mean_absolute_error(raw_tensor: torch.Tensor, predicted_tensor: torch.Tensor) -> float:
    """Compute mean absolute error between the raw tensor and the prediction."""

    return float((raw_tensor.float() - predicted_tensor.float()).abs().mean().item())


def summarize_prediction(raw_tensor: torch.Tensor, predicted_tensor: torch.Tensor) -> TensorMetricReport:
    """Summarize how well a predicted tensor matches a raw tensor."""

    residual_tensor = raw_tensor.float() - predicted_tensor.float()
    raw_energy = mean_squared_energy(raw_tensor)
    residual_energy = mean_squared_energy(residual_tensor)
    residual_energy_ratio = residual_energy / raw_energy if raw_energy > 0 else float("inf")

    return TensorMetricReport(
        raw_energy=raw_energy,
        residual_energy=residual_energy,
        residual_energy_ratio=residual_energy_ratio,
        cosine_similarity=cosine_similarity(raw_tensor, predicted_tensor),
        mean_absolute_error=mean_absolute_error(raw_tensor, predicted_tensor),
    )
