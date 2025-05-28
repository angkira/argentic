# Advanced LLM Configuration

This document explains the advanced LLM configuration system in Argentic, which allows fine-tuning of model parameters for each provider.

## Overview

The advanced configuration system provides granular control over LLM parameters for each provider. Parameters are organized in provider-specific sections within the `config.yaml` file, allowing you to optimize performance, quality, and behavior for your specific use case.

## Configuration Structure

Each provider has its own parameter section in the format `{provider}_parameters`. For example:

```yaml
llm:
  provider: google_gemini
  google_gemini_parameters:
    temperature: 0.7
    top_p: 0.95
    # ... other parameters
```

## Provider-Specific Parameters

### Google Gemini (`google_gemini_parameters`)

#### Core Sampling Parameters

- **`temperature`** (float, default: 0.7): Controls randomness. Higher values (0.8-1.0) make output more creative, lower values (0.1-0.3) make it more focused.
- **`top_p`** (float, default: 0.95): Nucleus sampling. Controls diversity by considering tokens with cumulative probability up to this value.
- **`top_k`** (int, default: 40): Limits next token selection to K most probable tokens.
- **`max_output_tokens`** (int, default: 2048): Maximum number of tokens to generate.
- **`candidate_count`** (int, default: 1): Number of response candidates to generate. Range: 1-8.
  - **Note**: Values > 1 generate multiple responses but only the first is returned by the provider.
  - **Recommendation**: Keep at 1 unless you need multiple response options for comparison.
  - **Performance**: Higher values increase API costs and latency proportionally.

#### Control Parameters

- **`stop_sequences`** (list, default: []): List of strings that will stop generation when encountered.

#### Safety and Content Filtering

- **`safety_settings`** (list, default: []): Configure content filtering. Example:
  ```yaml
  safety_settings:
    - category: HARM_CATEGORY_HARASSMENT
      threshold: BLOCK_MEDIUM_AND_ABOVE
    - category: HARM_CATEGORY_HATE_SPEECH
      threshold: BLOCK_MEDIUM_AND_ABOVE
  ```

#### Structured Output

- **`response_mime_type`** (string, optional): Set to "application/json" for JSON responses.
- **`response_schema`** (object, optional): JSON schema for structured output validation.

### Ollama (`ollama_parameters`)

#### Core Sampling Parameters

- **`temperature`** (float, default: 0.7): Controls randomness in generation.
- **`top_p`** (float, default: 0.9): Nucleus sampling threshold.
- **`top_k`** (int, default: 40): Top-k sampling limit.
- **`num_predict`** (int, default: 128): Maximum number of tokens to predict.
- **`repeat_penalty`** (float, default: 1.1): Penalty for repeating tokens (1.0 = no penalty).
- **`repeat_last_n`** (int, default: 64): Number of previous tokens to consider for repeat penalty.

#### Advanced Sampling

- **`tfs_z`** (float, default: 1.0): Tail free sampling parameter (1.0 = disabled).
- **`typical_p`** (float, default: 1.0): Locally typical sampling (1.0 = disabled).
- **`presence_penalty`** (float, default: 0.0): Penalty for token presence.
- **`frequency_penalty`** (float, default: 0.0): Penalty based on token frequency.

#### Context and Performance

- **`num_ctx`** (int, default: 2048): Context window size.
- **`num_batch`** (int, default: 512): Batch size for processing.
- **`num_gpu`** (int, default: 0): Number of GPU layers to use.
- **`main_gpu`** (int, default: 0): Main GPU to use.
- **`num_thread`** (int, default: -1): Number of threads (-1 for auto).

#### Control Parameters

- **`seed`** (int, default: -1): Random seed (-1 for random).
- **`stop`** (list, default: []): Stop sequences.

#### Performance Optimizations

- **`numa`** (bool, default: false): Enable NUMA optimizations.
- **`use_mmap`** (bool, default: true): Use memory mapping.
- **`use_mlock`** (bool, default: false): Lock memory pages.

### llama.cpp Server (`llama_cpp_server_parameters`)

#### Core Sampling Parameters

- **`temperature`** (float, default: 0.8): Sampling temperature.
- **`top_k`** (int, default: 40): Top-k sampling.
- **`top_p`** (float, default: 0.95): Nucleus sampling.
- **`min_p`** (float, default: 0.05): Minimum probability threshold.
- **`n_predict`** (int, default: 128): Number of tokens to predict.
- **`repeat_penalty`** (float, default: 1.1): Repetition penalty.
- **`repeat_last_n`** (int, default: 64): Tokens to consider for repetition penalty.

#### Advanced Sampling

- **`tfs_z`** (float, default: 1.0): Tail free sampling.
- **`typical_p`** (float, default: 1.0): Locally typical sampling.
- **`presence_penalty`** (float, default: 0.0): Presence penalty.
- **`frequency_penalty`** (float, default: 0.0): Frequency penalty.

#### Mirostat Sampling

- **`mirostat`** (int, default: 0): Mirostat mode (0=disabled, 1=Mirostat, 2=Mirostat 2.0).
- **`mirostat_tau`** (float, default: 5.0): Mirostat target entropy.
- **`mirostat_eta`** (float, default: 0.1): Mirostat learning rate.

#### Context Management

- **`n_ctx`** (int, default: 2048): Context size.
- **`n_keep`** (int, default: 0): Tokens to keep when context is full.
- **`n_batch`** (int, default: 512): Batch size.
- **`cache_prompt`** (bool, default: false): Enable prompt caching.

#### Control Parameters

- **`seed`** (int, default: -1): Random seed.
- **`stop`** (list, default: []): Stop sequences.
- **`ignore_eos`** (bool, default: false): Ignore end-of-sequence tokens.
- **`penalize_nl`** (bool, default: true): Penalize newline tokens.

#### Performance

- **`n_threads`** (int, default: -1): Number of threads.
- **`n_gpu_layers`** (int, default: 0): GPU layers to offload.

### llama.cpp CLI (`llama_cpp_cli_parameters`)

Parameters are automatically converted to command-line arguments:

#### Core Sampling

- **`temperature`** → `--temp`
- **`top_k`** → `--top-k`
- **`top_p`** → `--top-p`
- **`repeat_penalty`** → `--repeat-penalty`

#### Context and Performance

- **`ctx_size`** → `--ctx-size`
- **`batch_size`** → `--batch-size`
- **`threads`** → `--threads`
- **`n_gpu_layers`** → `--n-gpu-layers`

#### Control

- **`seed`** → `--seed`
- **`n_predict`** → `--n-predict`

#### Performance Optimizations

- **`mlock`** → `--mlock` (flag)
- **`no_mmap`** → `--no-mmap` (flag)

### llama.cpp Langchain (`llama_cpp_langchain_parameters`)

For the Langchain integration:

- **`temperature`** (float, default: 0.7)
- **`max_tokens`** (int, default: 256)
- **`top_p`** (float, default: 0.95)
- **`top_k`** (int, default: 40)
- **`repeat_penalty`** (float, default: 1.1)
- **`n_ctx`** (int, default: 2048)
- **`n_batch`** (int, default: 8)
- **`n_threads`** (int, default: -1)
- **`n_gpu_layers`** (int, default: 0)
- **`f16_kv`** (bool, default: true)
- **`use_mlock`** (bool, default: false)
- **`use_mmap`** (bool, default: true)
- **`verbose`** (bool, default: false)

## Parameter Impact on Performance and Quality

### Speed vs Quality Trade-offs

#### For Faster Responses:

- Lower `temperature` (0.1-0.3)
- Lower `top_k` (10-20)
- Lower `top_p` (0.7-0.8)
- Smaller `n_predict`/`max_tokens`
- Smaller `n_ctx`/`ctx_size`

#### For Higher Quality:

- Moderate `temperature` (0.7-0.9)
- Higher `top_k` (40-100)
- Higher `top_p` (0.9-0.95)
- Larger context windows
- Enable `cache_prompt` for repeated queries

#### For Creative Output:

- Higher `temperature` (0.8-1.2)
- Higher `top_p` (0.95-1.0)
- Lower `repeat_penalty` (1.0-1.05)
- Disable or reduce `frequency_penalty`

#### For Factual/Deterministic Output:

- Lower `temperature` (0.1-0.5)
- Lower `top_p` (0.7-0.9)
- Higher `repeat_penalty` (1.1-1.3)
- Set specific `seed` for reproducibility

### GPU Acceleration

For providers supporting GPU acceleration:

- **`n_gpu_layers`**: Start with small values (10-20) and increase
- **`num_gpu`**: Set to number of available GPUs
- **`main_gpu`**: Specify primary GPU for multi-GPU setups

### Memory Optimization

- **`use_mmap`**: Enable for large models to reduce RAM usage
- **`use_mlock`**: Enable to prevent swapping (requires sufficient RAM)
- **`f16_kv`**: Use 16-bit precision for key-value cache to save memory

## Example Configurations

### High-Performance Setup (Speed Priority)

```yaml
llm:
  provider: llama_cpp_server
  llama_cpp_server_parameters:
    temperature: 0.3
    top_k: 20
    top_p: 0.8
    n_predict: 64
    n_ctx: 1024
    n_gpu_layers: 35
    cache_prompt: true
```

### High-Quality Setup (Quality Priority)

```yaml
llm:
  provider: google_gemini
  google_gemini_parameters:
    temperature: 0.7
    top_p: 0.95
    top_k: 40
    max_output_tokens: 4096
    safety_settings: []
```

### Creative Writing Setup

```yaml
llm:
  provider: ollama
  ollama_parameters:
    temperature: 0.9
    top_p: 0.95
    top_k: 60
    repeat_penalty: 1.05
    presence_penalty: 0.1
    frequency_penalty: 0.1
```

### Deterministic/Factual Setup

```yaml
llm:
  provider: llama_cpp_server
  llama_cpp_server_parameters:
    temperature: 0.2
    top_k: 10
    top_p: 0.7
    repeat_penalty: 1.2
    seed: 42
    mirostat: 2
    mirostat_tau: 3.0
```

## Best Practices

1. **Start with defaults**: Begin with the provided default values and adjust incrementally.

2. **Test systematically**: Change one parameter at a time to understand its impact.

3. **Monitor performance**: Use logging to track response times and quality.

4. **Provider-specific tuning**: Each provider may respond differently to the same parameters.

5. **Context size considerations**: Larger contexts improve coherence but increase memory usage and latency.

6. **GPU memory management**: Monitor GPU memory usage when increasing `n_gpu_layers`.

7. **Reproducibility**: Set a fixed `seed` for consistent results during testing.

8. **Safety settings**: Configure appropriate safety settings for production deployments.

## Troubleshooting

### Common Issues

1. **Out of memory errors**: Reduce `n_ctx`, `n_batch`, or `n_gpu_layers`.

2. **Slow responses**: Increase `n_gpu_layers`, reduce context size, or lower quality parameters.

3. **Poor quality output**: Increase `temperature`, `top_p`, or context size.

4. **Repetitive output**: Increase `repeat_penalty` or `frequency_penalty`.

5. **Inconsistent results**: Set a fixed `seed` or adjust sampling parameters.

### Parameter Validation

The system validates parameters and will log warnings for:

- Values outside recommended ranges
- Incompatible parameter combinations
- Provider-specific limitations

Check the logs for parameter validation messages and adjust accordingly.
