"""Offline codec helpers for Phase 2 KV coding experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .metrics import cosine_similarity, mean_absolute_error, mean_squared_energy


RAW_DTYPE_BYTES = 2
BIT_WIDTH_TO_BYTES_PER_VALUE = {
    16: 2.0,
    8: 1.0,
    6: 0.75,
    4: 0.5,
    3: 0.375,
}


@dataclass(frozen=True)
class TensorCodecResult:
    """Reconstruction and byte estimates for one tensor under one codec."""

    codec_name: str
    quantized_values: torch.Tensor
    scales: torch.Tensor
    reconstructed: torch.Tensor
    mse: float
    cosine_similarity: float
    raw_bytes: int
    codec_bytes: float
    anchor_bytes: int
    payload_bytes: float
    scale_bytes: int


def _validate_kv_tensor_shape(tensor: torch.Tensor) -> None:
    """Require `[batch, kv_heads, seq_len, head_dim]` tensors."""

    if tensor.ndim != 4:
        raise ValueError(
            "Expected tensor shape [batch, kv_heads, seq_len, head_dim], "
            f"but observed {tuple(tensor.shape)}."
        )


def _validate_bit_width(bit_width: int) -> None:
    if bit_width not in BIT_WIDTH_TO_BYTES_PER_VALUE:
        raise ValueError(f"Unsupported bit width {bit_width}. Expected one of {sorted(BIT_WIDTH_TO_BYTES_PER_VALUE)}.")


def _quantization_qmax(bit_width: int) -> int:
    return (1 << (bit_width - 1)) - 1


def _quantize_block(block: torch.Tensor, bit_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one block with a symmetric scale per batch/head pair."""

    qmax = _quantization_qmax(bit_width)
    max_abs = block.abs().amax(dim=(-2, -1))
    scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    quantized = torch.round(block / scale.unsqueeze(-1).unsqueeze(-1)).clamp(-qmax, qmax).to(torch.int8)
    return quantized, scale.to(torch.float32)


def _payload_bytes(num_values: int, bit_width: int) -> float:
    return num_values * BIT_WIDTH_TO_BYTES_PER_VALUE[bit_width]


def _raw_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * RAW_DTYPE_BYTES


def _make_identity_codec_result(tensor: torch.Tensor, codec_name: str) -> TensorCodecResult:
    raw_bytes = _raw_bytes(tensor)
    return TensorCodecResult(
        codec_name=codec_name,
        quantized_values=torch.empty_like(tensor[..., :0, :], dtype=torch.int8),
        scales=torch.empty((tensor.shape[0], tensor.shape[1], 0), dtype=torch.float32, device=tensor.device),
        reconstructed=tensor.float().clone(),
        mse=0.0,
        cosine_similarity=1.0,
        raw_bytes=raw_bytes,
        codec_bytes=float(raw_bytes),
        anchor_bytes=0,
        payload_bytes=float(raw_bytes),
        scale_bytes=0,
    )


def _make_raw_fp16_codec_result(tensor: torch.Tensor, codec_name: str) -> TensorCodecResult:
    """Treat the original tensor as FP16-stored raw data."""

    raw_bytes = _raw_bytes(tensor)
    return TensorCodecResult(
        codec_name=codec_name,
        quantized_values=torch.empty_like(tensor[..., :0, :], dtype=torch.int8),
        scales=torch.empty((tensor.shape[0], tensor.shape[1], 0), dtype=torch.float32, device=tensor.device),
        reconstructed=tensor.float().clone(),
        mse=0.0,
        cosine_similarity=1.0,
        raw_bytes=raw_bytes,
        codec_bytes=float(raw_bytes),
        anchor_bytes=0,
        payload_bytes=float(raw_bytes),
        scale_bytes=0,
    )


def _build_previous_token_residuals(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Store token 0 raw and code deltas against the previous token."""

    anchor = tensor[..., :1, :].clone()
    residuals = tensor[..., 1:, :] - tensor[..., :-1, :]
    return anchor, residuals


def _reconstruct_previous_token(anchor: torch.Tensor, dequantized_residuals: torch.Tensor) -> torch.Tensor:
    reconstructed = torch.empty(
        (*anchor.shape[:-2], anchor.shape[-2] + dequantized_residuals.shape[-2], anchor.shape[-1]),
        dtype=torch.float32,
        device=anchor.device,
    )
    reconstructed[..., :1, :] = anchor.float()
    if dequantized_residuals.shape[-2] > 0:
        reconstructed[..., 1:, :] = anchor.float() + torch.cumsum(dequantized_residuals, dim=-2)
    return reconstructed


def _build_anchor_group_residuals(
    tensor: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor], list[tuple[int, int]]]:
    """Store one raw anchor per group and residuals only for non-anchor tokens."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")

    seq_len = tensor.shape[-2]
    num_groups = math.ceil(seq_len / group_size)
    anchors = torch.empty((*tensor.shape[:-2], num_groups, tensor.shape[-1]), dtype=tensor.dtype, device=tensor.device)
    residual_blocks: list[torch.Tensor] = []
    spans: list[tuple[int, int]] = []

    for group_index, start in enumerate(range(0, seq_len, group_size)):
        end = min(start + group_size, seq_len)
        spans.append((start, end))
        anchor = tensor[..., start:start + 1, :]
        anchors[..., group_index:group_index + 1, :] = anchor
        residual_blocks.append(tensor[..., start + 1:end, :] - anchor)

    return anchors, residual_blocks, spans


def _reconstruct_anchor_group(
    anchors: torch.Tensor,
    dequantized_residual_blocks: list[torch.Tensor],
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    seq_len = spans[-1][1] if spans else 0
    reconstructed = torch.empty(
        (*anchors.shape[:-2], seq_len, anchors.shape[-1]),
        dtype=torch.float32,
        device=anchors.device,
    )
    for group_index, (start, end) in enumerate(spans):
        reconstructed[..., start:start + 1, :] = anchors[..., group_index:group_index + 1, :].float()
        if end - start > 1:
            reconstructed[..., start + 1:end, :] = (
                anchors[..., group_index:group_index + 1, :].float() + dequantized_residual_blocks[group_index]
            )
    return reconstructed


def _apply_block_quantization(
    residuals: torch.Tensor,
    bit_width: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a token-axis signal blockwise with one symmetric scale per head and block."""

    _validate_bit_width(bit_width)
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    batch_size, num_heads, seq_len, _ = residuals.shape
    num_blocks = math.ceil(seq_len / group_size) if seq_len > 0 else 0
    quantized = torch.empty_like(residuals, dtype=torch.int8)
    scales = torch.empty((batch_size, num_heads, num_blocks), dtype=torch.float32, device=residuals.device)
    dequantized = torch.empty_like(residuals, dtype=torch.float32)

    for block_index in range(num_blocks):
        start = block_index * group_size
        end = min(start + group_size, seq_len)
        block = residuals[..., start:end, :]
        block_quantized, block_scale = _quantize_block(block, bit_width)
        quantized[..., start:end, :] = block_quantized
        scales[..., block_index] = block_scale
        dequantized[..., start:end, :] = block_quantized.float() * block_scale.unsqueeze(-1).unsqueeze(-1)

    return quantized, scales, dequantized


def encode_previous_token_residual(
    tensor: torch.Tensor,
    bit_width: int,
    group_size: int,
    codec_name: str,
) -> TensorCodecResult:
    """Code token 0 raw and the remaining signal as previous-token residuals."""

    _validate_kv_tensor_shape(tensor)
    if tensor.shape[-2] == 0:
        raise ValueError("Tensor must contain at least one token.")

    anchor, residuals = _build_previous_token_residuals(tensor)
    anchor_bytes = anchor.numel() * RAW_DTYPE_BYTES

    if bit_width == 16:
        quantized = torch.empty_like(residuals, dtype=torch.int8)
        scales = torch.empty((tensor.shape[0], tensor.shape[1], 0), dtype=torch.float32, device=tensor.device)
        dequantized = residuals.float()
        payload_bytes = _payload_bytes(residuals.numel(), bit_width)
        scale_bytes = 0
    else:
        quantized, scales, dequantized = _apply_block_quantization(residuals, bit_width=bit_width, group_size=group_size)
        payload_bytes = _payload_bytes(quantized.numel(), bit_width)
        scale_bytes = scales.numel() * 4

    reconstructed = _reconstruct_previous_token(anchor, dequantized)
    codec_bytes = anchor_bytes + payload_bytes + scale_bytes

    return TensorCodecResult(
        codec_name=codec_name,
        quantized_values=quantized,
        scales=scales,
        reconstructed=reconstructed,
        mse=mean_squared_energy(tensor.float() - reconstructed),
        cosine_similarity=cosine_similarity(tensor, reconstructed),
        raw_bytes=_raw_bytes(tensor),
        codec_bytes=codec_bytes,
        anchor_bytes=anchor_bytes,
        payload_bytes=payload_bytes,
        scale_bytes=scale_bytes,
    )


def encode_anchor_group_residual(
    tensor: torch.Tensor,
    bit_width: int,
    group_size: int,
    codec_name: str,
) -> TensorCodecResult:
    """Code one raw anchor per group and residuals relative to that anchor."""

    _validate_kv_tensor_shape(tensor)
    anchors, residual_blocks, spans = _build_anchor_group_residuals(tensor, group_size=group_size)
    quantized_blocks: list[torch.Tensor] = []
    dequantized_blocks: list[torch.Tensor] = []
    scale_values: list[torch.Tensor] = []

    for residual_block in residual_blocks:
        if residual_block.shape[-2] == 0:
            quantized_blocks.append(torch.empty_like(residual_block, dtype=torch.int8))
            dequantized_blocks.append(torch.empty_like(residual_block, dtype=torch.float32))
            if bit_width != 16:
                scale_values.append(torch.ones((*residual_block.shape[:2], 1), dtype=torch.float32, device=tensor.device))
            continue

        if bit_width == 16:
            quantized_blocks.append(torch.empty_like(residual_block, dtype=torch.int8))
            dequantized_blocks.append(residual_block.float())
        else:
            quantized_block, scales_block, dequantized_block = _apply_block_quantization(
                residual_block,
                bit_width=bit_width,
                group_size=max(1, residual_block.shape[-2]),
            )
            quantized_blocks.append(quantized_block)
            dequantized_blocks.append(dequantized_block)
            scale_values.append(scales_block)

    quantized = (
        torch.cat(quantized_blocks, dim=-2)
        if quantized_blocks
        else torch.empty_like(tensor[..., :0, :], dtype=torch.int8)
    )
    scales = (
        torch.cat(scale_values, dim=-1)
        if scale_values
        else torch.empty((tensor.shape[0], tensor.shape[1], 0), dtype=torch.float32, device=tensor.device)
    )
    reconstructed = _reconstruct_anchor_group(anchors, dequantized_blocks, spans)

    anchor_bytes = anchors.numel() * RAW_DTYPE_BYTES
    residual_numel = sum(block.numel() for block in residual_blocks)
    payload_bytes = _payload_bytes(residual_numel, bit_width)
    scale_bytes = scales.numel() * 4 if bit_width != 16 else 0
    codec_bytes = anchor_bytes + payload_bytes + scale_bytes

    return TensorCodecResult(
        codec_name=codec_name,
        quantized_values=quantized,
        scales=scales,
        reconstructed=reconstructed,
        mse=mean_squared_energy(tensor.float() - reconstructed),
        cosine_similarity=cosine_similarity(tensor, reconstructed),
        raw_bytes=_raw_bytes(tensor),
        codec_bytes=codec_bytes,
        anchor_bytes=anchor_bytes,
        payload_bytes=payload_bytes,
        scale_bytes=scale_bytes,
    )


def _build_dct_matrix(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create an orthonormal DCT-II matrix for a given block length."""

    indices_n = torch.arange(length, device=device, dtype=torch.float32)
    indices_k = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    matrix = torch.cos(math.pi / length * (indices_n + 0.5) * indices_k)
    matrix[0, :] = matrix[0, :] * math.sqrt(1.0 / length)
    if length > 1:
        matrix[1:, :] = matrix[1:, :] * math.sqrt(2.0 / length)
    return matrix.to(dtype=dtype)


def encode_block_dct(
    tensor: torch.Tensor,
    bit_width: int,
    group_size: int,
    codec_name: str,
) -> TensorCodecResult:
    """Apply block DCT over the token axis, then quantize all coefficients."""

    _validate_kv_tensor_shape(tensor)
    _validate_bit_width(bit_width)
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    batch_size, num_heads, seq_len, head_dim = tensor.shape
    quantized = torch.empty_like(tensor, dtype=torch.int8)
    reconstructed = torch.empty_like(tensor, dtype=torch.float32)
    num_blocks = math.ceil(seq_len / group_size) if seq_len > 0 else 0
    scales = torch.empty((batch_size, num_heads, num_blocks), dtype=torch.float32, device=tensor.device)

    for block_index, start in enumerate(range(0, seq_len, group_size)):
        end = min(start + group_size, seq_len)
        block = tensor[..., start:end, :].float()
        dct_matrix = _build_dct_matrix(end - start, device=tensor.device, dtype=block.dtype)

        coefficients = torch.einsum("ij,bhjd->bhid", dct_matrix, block)
        block_quantized, block_scale = _quantize_block(coefficients, bit_width)
        scales[..., block_index] = block_scale
        quantized[..., start:end, :] = block_quantized

        dequantized_coefficients = block_quantized.float() * block_scale.unsqueeze(-1).unsqueeze(-1)
        reconstructed[..., start:end, :] = torch.einsum("ji,bhid->bhjd", dct_matrix, dequantized_coefficients)

    payload_bytes = _payload_bytes(quantized.numel(), bit_width)
    scale_bytes = scales.numel() * 4
    codec_bytes = payload_bytes + scale_bytes

    return TensorCodecResult(
        codec_name=codec_name,
        quantized_values=quantized,
        scales=scales,
        reconstructed=reconstructed,
        mse=mean_squared_energy(tensor.float() - reconstructed),
        cosine_similarity=cosine_similarity(tensor, reconstructed),
        raw_bytes=_raw_bytes(tensor),
        codec_bytes=codec_bytes,
        anchor_bytes=0,
        payload_bytes=payload_bytes,
        scale_bytes=scale_bytes,
    )


def encode_raw_baseline(tensor: torch.Tensor, codec_name: str = "raw") -> TensorCodecResult:
    """Treat the original tensor as the reconstructed tensor and keep raw bytes."""

    _validate_kv_tensor_shape(tensor)
    return _make_identity_codec_result(tensor, codec_name=codec_name)


def encode_raw_tensor(
    tensor: torch.Tensor,
    bit_width: int,
    group_size: int,
    codec_name: str,
) -> TensorCodecResult:
    """Store the tensor directly, either as FP16 or as quantized packed values."""

    _validate_kv_tensor_shape(tensor)
    _validate_bit_width(bit_width)

    if bit_width == 16:
        return _make_raw_fp16_codec_result(tensor, codec_name=codec_name)

    quantized, scales, dequantized = _apply_block_quantization(tensor, bit_width=bit_width, group_size=group_size)
    payload_bytes = _payload_bytes(quantized.numel(), bit_width)
    scale_bytes = scales.numel() * 4
    codec_bytes = payload_bytes + scale_bytes

    return TensorCodecResult(
        codec_name=codec_name,
        quantized_values=quantized,
        scales=scales,
        reconstructed=dequantized,
        mse=mean_squared_energy(tensor.float() - dequantized),
        cosine_similarity=cosine_similarity(tensor, dequantized),
        raw_bytes=_raw_bytes(tensor),
        codec_bytes=codec_bytes,
        anchor_bytes=0,
        payload_bytes=payload_bytes,
        scale_bytes=scale_bytes,
    )


def qk_logit_mae(last_query: torch.Tensor, raw_k: torch.Tensor, reconstructed_k: torch.Tensor, scaling: float) -> float:
    """Compare raw and reconstructed attention logits for the last token query."""

    raw_logits = torch.matmul(last_query.float(), raw_k.float().transpose(-1, -2)) * scaling
    reconstructed_logits = torch.matmul(last_query.float(), reconstructed_k.float().transpose(-1, -2)) * scaling
    return mean_absolute_error(raw_logits, reconstructed_logits)
