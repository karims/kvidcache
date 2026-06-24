"""Run Phase 3B short generation replay with teacher-forced compressed KV."""

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
from kvidcache.codec import encode_anchor_group_residual, encode_previous_token_residual, encode_raw_tensor
from kvidcache.metrics import cosine_similarity


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model name or path.")
    parser.add_argument("--prompt-file", default="prompts/code_prompt.txt", help="Prompt file to analyze.")
    parser.add_argument("--k-codec", default="prev-residual", choices=["raw", "prev-residual"], help="K codec to use.")
    parser.add_argument(
        "--v-codec",
        default="anchor-residual",
        choices=["raw", "anchor-residual"],
        help="V codec to use.",
    )
    parser.add_argument("--k-bits", type=int, default=8, choices=[16, 8, 6, 4], help="Quantization bit width for K.")
    parser.add_argument("--v-bits", type=int, default=4, choices=[16, 8, 6, 4], help="Quantization bit width for V.")
    parser.add_argument("--group-size", type=int, default=8, choices=[4, 8, 16, 32, 64, 128], help="Token group size for the codecs.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, for example `cpu` or `cuda`.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of prompt tokens to keep.")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Number of greedy continuation steps to test.")
    parser.add_argument(
        "--mode",
        default="teacher-forced",
        choices=["teacher-forced", "free-running"],
        help="Whether compressed replay uses raw next tokens or feeds back its own greedy tokens.",
    )
    return parser


def tokenize_prompt(tokenizer, prompt: str, max_tokens: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the tokenizer chat template and tokenize the prompt."""

    formatted_prompt = prepare_prompt_text(tokenizer, prompt)
    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def get_peak_cuda_memory_allocated(device: torch.device) -> int:
    """Return peak allocated CUDA memory in bytes for the selected device."""

    if device.type != "cuda":
        return 0

    return int(torch.cuda.max_memory_allocated(device))


def compress_prefix_cache(
    prefix_past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    k_codec: str,
    v_codec: str,
    k_bits: int,
    v_bits: int,
    group_size: int,
) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], tuple[float, float, float, float]], ...]:
    """Reconstruct prefix KV and return per-layer raw/codec byte estimates."""

    reconstructed_layers = []
    for raw_k, raw_v in prefix_past_key_values:
        if k_codec == "raw":
            k_result = encode_raw_tensor(
                raw_k,
                bit_width=k_bits,
                group_size=group_size,
                codec_name=f"k-raw-int{k_bits}",
            )
        elif k_codec == "prev-residual":
            k_result = encode_previous_token_residual(
                raw_k,
                bit_width=k_bits,
                group_size=group_size,
                codec_name=f"k-{k_codec}-int{k_bits}",
            )
        else:
            raise ValueError(f"Unsupported K codec: {k_codec}")

        if v_codec == "raw":
            v_result = encode_raw_tensor(
                raw_v,
                bit_width=v_bits,
                group_size=group_size,
                codec_name=f"v-raw-int{v_bits}",
            )
        elif v_codec == "anchor-residual":
            v_result = encode_anchor_group_residual(
                raw_v,
                bit_width=v_bits,
                group_size=group_size,
                codec_name=f"v-{v_codec}-int{v_bits}",
            )
        else:
            raise ValueError(f"Unsupported V codec: {v_codec}")
        reconstructed_layers.append(
            (
                (
                    k_result.reconstructed.detach().to(dtype=raw_k.dtype),
                    v_result.reconstructed.detach().to(dtype=raw_v.dtype),
                ),
                (
                    float(k_result.raw_bytes),
                    float(k_result.codec_bytes),
                    float(v_result.raw_bytes),
                    float(v_result.codec_bytes),
                ),
            )
        )
    return tuple(reconstructed_layers)


def run_attention_with_prefix(attention, hidden_states: torch.Tensor, position_embeddings, past_k: torch.Tensor, past_v: torch.Tensor) -> torch.Tensor:
    """Run one attention step for a single token against reconstructed prefix KV."""

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
    current_token_id: torch.Tensor,
    reconstructed_prefix_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    prefix_length: int,
) -> torch.Tensor:
    """Replay one token through the decoder using reconstructed prefix KV."""

    with torch.inference_mode():
        hidden_states = model.model.embed_tokens(current_token_id)
        position_ids = torch.tensor([[prefix_length]], device=current_token_id.device)
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

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
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = model.model.norm(hidden_states)
        return model.lm_head(hidden_states).detach()


def topk_overlap_count(raw_logits: torch.Tensor, replay_logits: torch.Tensor, k: int) -> int:
    """Count overlap between the top-k token sets."""

    raw_topk = set(torch.topk(raw_logits[0, -1], k=k).indices.tolist())
    replay_topk = set(torch.topk(replay_logits[0, -1], k=k).indices.tolist())
    return len(raw_topk.intersection(replay_topk))


def raw_greedy_generate(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int) -> tuple[list[int], list[torch.Tensor]]:
    """Generate a short raw greedy continuation and capture per-step logits."""

    generated_tokens: list[int] = []
    step_logits_list: list[torch.Tensor] = []
    current_ids = input_ids
    current_mask = attention_mask

    for step_index in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_mask,
                use_cache=False,
            )

        step_logits_cpu = outputs.logits[:, -1, :].detach().float().cpu()
        next_token = int(torch.argmax(step_logits_cpu[0]).item())
        generated_tokens.append(next_token)
        step_logits_list.append(step_logits_cpu)

        next_token_tensor = torch.tensor([[next_token]], device=current_ids.device)
        current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
        current_mask = torch.cat([current_mask, torch.ones_like(next_token_tensor)], dim=1)

        del outputs, step_logits_cpu, next_token_tensor
        if current_ids.device.type == "cuda" and (step_index + 1) % 25 == 0:
            torch.cuda.empty_cache()

    return generated_tokens, step_logits_list


def compressed_replay_generate(
    model,
    prompt_ids: torch.Tensor,
    raw_generated_tokens: list[int],
    k_codec: str,
    v_codec: str,
    k_bits: int,
    v_bits: int,
    group_size: int,
    mode: str,
) -> tuple[list[int], list[torch.Tensor]]:
    """Run compressed replay in teacher-forced or free-running mode."""

    replay_tokens: list[int] = []
    replay_logits_per_step: list[torch.Tensor] = []

    context_ids = prompt_ids.clone()
    for step_index, raw_token in enumerate(raw_generated_tokens):
        if context_ids.shape[-1] < 2:
            raise ValueError("Need at least two context tokens for one-step replay.")

        prefix_ids = context_ids[:, :-1]
        current_token_id = context_ids[:, -1:]
        prefix_mask = torch.ones_like(prefix_ids)

        with torch.inference_mode():
            prefix_outputs = model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                use_cache=True,
            )

            reconstructed_prefix_kv = compress_prefix_cache(
                normalize_past_key_values(prefix_outputs.past_key_values),
                k_codec=k_codec,
                v_codec=v_codec,
                k_bits=k_bits,
                v_bits=v_bits,
                group_size=group_size,
            )
            replay_logits = replay_one_step_with_reconstructed_prefix(
                model=model,
                current_token_id=current_token_id,
                reconstructed_prefix_kv=tuple(layer_pair for layer_pair, _ in reconstructed_prefix_kv),
                prefix_length=int(prefix_ids.shape[-1]),
            )

        replay_step_logits = replay_logits[:, -1, :].detach().float().cpu()
        replay_token = int(torch.argmax(replay_step_logits[0]).item())
        replay_tokens.append(replay_token)
        replay_logits_per_step.append(replay_step_logits)

        if mode == "teacher-forced":
            next_token = raw_token
        elif mode == "free-running":
            next_token = replay_token
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        next_token_tensor = torch.tensor([[next_token]], device=context_ids.device)
        context_ids = torch.cat([context_ids, next_token_tensor], dim=1)

        del prefix_outputs, reconstructed_prefix_kv, replay_logits, replay_step_logits, prefix_ids, current_token_id, prefix_mask, next_token_tensor
        if context_ids.device.type == "cuda" and (step_index + 1) % 25 == 0:
            torch.cuda.empty_cache()

    return replay_tokens, replay_logits_per_step


def find_first_divergence(raw_tokens: list[int], replay_tokens: list[int]) -> int | None:
    """Return the first 1-based generation step where tokens diverge."""

    for index, (raw_token, replay_token) in enumerate(zip(raw_tokens, replay_tokens), start=1):
        if raw_token != replay_token:
            return index
    return None


def main() -> int:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"

    prompt = load_prompt(args.prompt_file)
    input_ids, attention_mask = tokenize_prompt(tokenizer, prompt, args.max_tokens, device)
    if input_ids.shape[-1] < 2:
        raise ValueError("Need at least two prompt tokens for teacher-forced replay.")

    with torch.inference_mode():
        prompt_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    prompt_codec_layers = compress_prefix_cache(
        normalize_past_key_values(prompt_outputs.past_key_values),
        k_codec=args.k_codec,
        v_codec=args.v_codec,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        group_size=args.group_size,
    )
    total_raw_k_bytes = sum(layer_bytes[0] for _, layer_bytes in prompt_codec_layers)
    total_codec_k_bytes = sum(layer_bytes[1] for _, layer_bytes in prompt_codec_layers)
    total_raw_v_bytes = sum(layer_bytes[2] for _, layer_bytes in prompt_codec_layers)
    total_codec_v_bytes = sum(layer_bytes[3] for _, layer_bytes in prompt_codec_layers)
    estimated_full_kv_compression_ratio = (total_raw_k_bytes + total_raw_v_bytes) / (
        total_codec_k_bytes + total_codec_v_bytes
    )

    raw_generated_tokens, raw_logits_per_step = raw_greedy_generate(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
    )
    replay_generated_tokens, replay_logits_per_step = compressed_replay_generate(
        model=model,
        prompt_ids=input_ids,
        raw_generated_tokens=raw_generated_tokens,
        k_codec=args.k_codec,
        v_codec=args.v_codec,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        group_size=args.group_size,
        mode=args.mode,
    )

    token_match_count = sum(int(raw_token == replay_token) for raw_token, replay_token in zip(raw_generated_tokens, replay_generated_tokens))
    token_match_rate = token_match_count / len(raw_generated_tokens) if raw_generated_tokens else 0.0
    first_divergence = find_first_divergence(raw_generated_tokens, replay_generated_tokens)

    logit_cosines = [
        cosine_similarity(raw_logits, replay_logits)
        for raw_logits, replay_logits in zip(raw_logits_per_step, replay_logits_per_step)
    ]
    average_logit_cosine = sum(logit_cosines) / len(logit_cosines) if logit_cosines else 0.0
    worst_logit_cosine = min(logit_cosines) if logit_cosines else 0.0
    top1_agreement_count = token_match_count
    top5_overlap_average = (
        sum(topk_overlap_count(raw_logits, replay_logits, k=5) / 5.0 for raw_logits, replay_logits in zip(raw_logits_per_step, replay_logits_per_step))
        / len(raw_logits_per_step)
        if raw_logits_per_step
        else 0.0
    )

    raw_text = tokenizer.decode(raw_generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    replay_text = tokenizer.decode(replay_generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    peak_cuda_memory_allocated = get_peak_cuda_memory_allocated(device)

    print(f"mode: {args.mode} compressed replay")
    print(f"model name: {args.model}")
    print(f"prompt token count: {input_ids.shape[-1]}")
    print(f"selected K codec: {args.k_codec} int{args.k_bits}")
    print(f"selected V codec: {args.v_codec} int{args.v_bits}")
    print(f"group size: {args.group_size}")
    print(f"max new tokens: {args.max_new_tokens}")
    print(f"estimated full KV compression ratio: {estimated_full_kv_compression_ratio:.4f}")
    print()
    print("raw generated text:")
    print(raw_text)
    print()
    print("compressed generated text:")
    print(replay_text)
    print()
    print(f"token match count: {token_match_count}")
    print(f"token match rate: {token_match_rate:.4f}")
    print(f"first divergence position: {first_divergence if first_divergence is not None else 'none'}")
    print(f"average logit cosine: {average_logit_cosine:.8f}")
    print(f"worst logit cosine: {worst_logit_cosine:.8f}")
    print(f"top-1 agreement count: {top1_agreement_count}")
    print(f"top-5 overlap average: {top5_overlap_average:.4f}")
    print(f"peak CUDA memory allocated: {peak_cuda_memory_allocated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
