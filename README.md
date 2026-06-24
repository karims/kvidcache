# kvidcache

`kvidcache` is a small PyTorch-only research repo for testing whether transformer KV cache has useful temporal redundancy along the token axis.

The core analogy is video compression:

- periodic token positions act like anchor frames
- nearby token positions are predicted from local context
- the stored signal becomes residuals instead of raw KV everywhere

If those residuals are consistently lower-energy than the original KV tensors, that is a useful proof-of-signal for future KV cache coding work.

## Phase 1

Phase 1 is intentionally narrow. It does not build an inference engine, custom kernels, or a serving stack.

It only tests whether simple token-axis predictors explain KV structure:

- previous-token predictor
- periodic anchor predictor
- linear extrapolation predictor

For each predictor, the repo captures `past_key_values` from a small decoder-only Hugging Face model and compares raw tensors against predicted tensors using:

- `raw_energy = mean(x^2)`
- `residual_energy = mean((x - x_hat)^2)`
- `residual_energy_ratio = residual_energy / raw_energy`
- cosine similarity
- mean absolute error

## Out Of Scope

- not an inference engine
- not a custom kernel project
- not a CUDA extension project
- not a vLLM or SGLang integration

## Layout

```text
kvidcache/
  README.md
  requirements.txt
  prompts/
    code_prompt.txt
  kvidcache/
    __init__.py
    capture.py
    predictors.py
    metrics.py
  scripts/
    run_residual_analysis.py
  outputs/
    .gitkeep
```

## How To Run

Install dependencies, then run:

```bash
python scripts/run_residual_analysis.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-file prompts/code_prompt.txt \
  --group-size 16 \
  --max-tokens 1024
```

Defaults:

- `--model Qwen/Qwen2.5-0.5B-Instruct`
- `--prompt-file prompts/code_prompt.txt`
- `--group-size 16`
- `--device cuda` if available, else `cpu`
- `--max-tokens 1024`

The script prints a summary table for both `K` and `V` with:

- average residual energy ratio across layers
- best layer residual energy ratio
- worst layer residual energy ratio
- average cosine similarity
- average mean absolute error

It also writes a JSON report to `outputs/residual_report.json`.

## Interpreting `residual_energy_ratio`

`residual_energy_ratio` is the main Phase 1 signal.

- a ratio below `1.0` means the residual is lower-energy than the raw KV tensor
- a smaller ratio is better
- a ratio near `0.0` suggests stronger temporal predictability along the token axis
- if `K` or `V` show consistently low ratios across layers, predictive KV coding may be promising

## Notes

- The first run may download the model from Hugging Face.
- CPU execution is supported, but it may be slower.
- The script expects KV tensors shaped like `[batch, kv_heads, seq_len, head_dim]` and raises a clear error if a model returns something different.
