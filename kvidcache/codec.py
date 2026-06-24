"""Small K-only predictive residual codec helpers for Phase 2A."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .metrics import cosine_similarity, mean_absolute_error, mean_squared_energy


@dataclass(frozen=True)
class KCodecLayerResult:
    """Codec outputs and metrics for one layer's K tensor."""

    anchor: torch.Tensor
    quantized_residuals: torch.Tensor
    scales: torch.Tensor
    reconstructed: torch.Tensor
    reconstruction_mse: float
    reconstruction_cosine_similarity: float
    raw_k_bytes: int
    codec_k_bytes: int
    scale_bytes: int


def _validate_k_tensor_shape(k_tensor: torch.Tensor) -> None:
    """Require `[batch, kv_heads, seq_len, head_dim]` for K tensors."""

    if k_tensor.ndim != 4:
        raise ValueError(
            "Expected K tensor shape [batch, kv_heads, seq_len, head_dim], "
            f"but observed {tuple(k_tensor.shape)}."
        )


def _quantize_residual_block(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one `[batch, heads, tokens, dim]` residual block to int8."""

    max_abs = block.abs().amax(dim=(-2, -1))
    scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
    quantized = torch.round(block / scale.unsqueeze(-1).unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return quantized, scale.to(torch.float32)


def encode_k_previous_token_int8(k_tensor: torch.Tensor, group_size: int) -> KCodecLayerResult:
    """Encode one K tensor with raw token 0 and int8 residuals for tokens 1..T-1."""

    _validate_k_tensor_shape(k_tensor)
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    batch_size, num_heads, seq_len, head_dim = k_tensor.shape
    if seq_len == 0:
        raise ValueError("K tensor must contain at least one token.")

    anchor = k_tensor[..., :1, :].clone()
    if seq_len == 1:
        reconstructed = anchor.clone()
        return KCodecLayerResult(
            anchor=anchor,
            quantized_residuals=torch.empty_like(k_tensor[..., :0, :], dtype=torch.int8),
            scales=torch.empty((batch_size, num_heads, 0), dtype=torch.float32, device=k_tensor.device),
            reconstructed=reconstructed,
            reconstruction_mse=0.0,
            reconstruction_cosine_similarity=1.0,
            raw_k_bytes=k_tensor.numel() * 2,
            codec_k_bytes=anchor.numel() * 2,
            scale_bytes=0,
        )

    residuals = k_tensor[..., 1:, :] - k_tensor[..., :-1, :]
    residual_seq_len = residuals.shape[-2]
    num_blocks = (residual_seq_len + group_size - 1) // group_size

    quantized_residuals = torch.empty_like(residuals, dtype=torch.int8)
    scales = torch.empty((batch_size, num_heads, num_blocks), dtype=torch.float32, device=k_tensor.device)

    for block_index in range(num_blocks):
        start = block_index * group_size
        end = min(start + group_size, residual_seq_len)
        block = residuals[..., start:end, :]
        block_quantized, block_scale = _quantize_residual_block(block)
        quantized_residuals[..., start:end, :] = block_quantized
        scales[..., block_index] = block_scale

    dequantized_residuals = torch.empty_like(residuals, dtype=torch.float32)
    for block_index in range(num_blocks):
        start = block_index * group_size
        end = min(start + group_size, residual_seq_len)
        scale = scales[..., block_index].unsqueeze(-1).unsqueeze(-1)
        dequantized_residuals[..., start:end, :] = quantized_residuals[..., start:end, :].float() * scale

    reconstructed = torch.empty_like(k_tensor, dtype=torch.float32)
    reconstructed[..., :1, :] = anchor.float()
    reconstructed[..., 1:, :] = anchor.float() + torch.cumsum(dequantized_residuals, dim=-2)

    scale_bytes = scales.numel() * 4
    raw_k_bytes = k_tensor.numel() * 2
    codec_k_bytes = anchor.numel() * 2 + quantized_residuals.numel() * 1 + scale_bytes

    return KCodecLayerResult(
        anchor=anchor,
        quantized_residuals=quantized_residuals,
        scales=scales,
        reconstructed=reconstructed,
        reconstruction_mse=mean_squared_energy(k_tensor.float() - reconstructed.float()),
        reconstruction_cosine_similarity=cosine_similarity(k_tensor, reconstructed),
        raw_k_bytes=raw_k_bytes,
        codec_k_bytes=codec_k_bytes,
        scale_bytes=scale_bytes,
    )


def qk_logit_mae(last_query: torch.Tensor, raw_k: torch.Tensor, reconstructed_k: torch.Tensor, scaling: float) -> float:
    """Compare raw and reconstructed attention logits for the last token query."""

    raw_logits = torch.matmul(last_query.float(), raw_k.float().transpose(-1, -2)) * scaling
    reconstructed_logits = torch.matmul(last_query.float(), reconstructed_k.float().transpose(-1, -2)) * scaling
    return mean_absolute_error(raw_logits, reconstructed_logits)
