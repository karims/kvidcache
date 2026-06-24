"""Toy predictors for token-axis KV residual analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


Predictor = Callable[[torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True)
class PredictorSpec:
    """A predictor plus a short display name."""

    name: str
    predictor: Predictor


def previous_token_predictor(tensor: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Predict each token from the immediately previous token.

    The first token is treated as its own anchor because there is no previous
    token available at the start of the sequence.
    """

    predicted = torch.empty_like(tensor)
    predicted[..., 0, :] = tensor[..., 0, :]
    predicted[..., 1:, :] = tensor[..., :-1, :]
    return predicted


def anchor_predictor(tensor: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Predict tokens from the most recent periodic anchor token.

    Every `group_size`-th token is treated as an exact anchor. Intermediate
    tokens are predicted by copying the anchor for that group.
    """

    if group_size <= 0:
        raise ValueError("group_size must be positive")

    predicted = torch.empty_like(tensor)
    seq_len = tensor.shape[-2]

    for start in range(0, seq_len, group_size):
        end = min(start + group_size, seq_len)
        predicted[..., start:end, :] = tensor[..., start:start + 1, :]

    return predicted


def linear_extrapolation_predictor(tensor: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Predict each token using a simple linear extrapolation.

    For the first predictable token we fall back to the previous-token rule:
    `x_hat[1] = x[0]`. After that we use
    `x_hat[t] = 2 * x[t-1] - x[t-2]`.
    """

    predicted = torch.empty_like(tensor)
    seq_len = tensor.shape[-2]

    if seq_len == 0:
        return predicted

    predicted[..., 0, :] = tensor[..., 0, :]
    if seq_len > 1:
        predicted[..., 1, :] = tensor[..., 0, :]

    for idx in range(2, seq_len):
        predicted[..., idx, :] = 2 * tensor[..., idx - 1, :] - tensor[..., idx - 2, :]

    return predicted


PREDICTORS: tuple[PredictorSpec, ...] = (
    PredictorSpec(name="previous-token", predictor=previous_token_predictor),
    PredictorSpec(name="anchor", predictor=anchor_predictor),
    PredictorSpec(name="linear-extrapolation", predictor=linear_extrapolation_predictor),
)
