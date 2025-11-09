# vLLM VRAM Issue - Solutions

## Problem
- RTX 4060 Ti: 16 GB total
- Model loaded: 15797 MB used (96.4%)
- **Only 583 MB free** - not enough for inference!

## Why Connections Timeout
The server loads successfully but when a request arrives:
1. vLLM needs ~1-2 GB for KV cache
2. Vision models need extra memory for image embeddings
3. Not enough free VRAM → request queues/hangs → timeout

## Solutions (Try in Order)

### Solution 1: Reduce KV Cache (Quickest)
```bash
# Kill current server
pkill -f vllm

# Restart with smaller KV cache
vllm serve google/gemma-3n-E2B-it \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt image=1
```

This reserves 15% VRAM for inference (~2.4 GB).

### Solution 2: Use Quantized Model (Best for RTX 4060 Ti 16GB)
Quantized models use 50% less VRAM!

```bash
# Use 4-bit quantized version
vllm serve google/gemma-2-9b-it \
  --port 8000 \
  --host 0.0.0.0 \
  --quantization awq \
  --limit-mm-per-prompt image=1
```

Or find a quantized version of Gemma 3n:
```bash
vllm serve google/gemma-3n-E2B-it \
  --port 8000 \
  --host 0.0.0.0 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --gpu-memory-utilization 0.80
```

### Solution 3: CPU Offloading
Offload some layers to CPU RAM:

```bash
vllm serve google/gemma-3n-E2B-it \
  --port 8000 \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt image=1
```

### Solution 4: Use Smaller Vision Model
Gemma 3n E2B might be too large. Try a smaller vision model:

```bash
# Gemma 2B (much smaller)
vllm serve google/gemma-2-2b-it \
  --port 8000 \
  --host 0.0.0.0 \
  --limit-mm-per-prompt image=1
```

Or use Qwen2-VL (optimized for vision):
```bash
vllm serve Qwen/Qwen2-VL-2B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 2048
```

### Solution 5: Close Other GPU Apps
Check what else is using GPU:
```bash
nvidia-smi
# Kill any unnecessary processes
```

## Recommended: Start with Solution 1

```bash
# 1. Kill current vLLM
pkill -f vllm
# Wait 5 seconds
sleep 5

# 2. Restart with memory optimization
vllm serve google/gemma-3n-E2B-it \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt image=1

# 3. Wait for "Application startup complete"

# 4. Test
curl http://localhost:8000/v1/models
```

Then test your visual agent:
```bash
python ./examples/visual_agent_vllm.py
```

## Monitor Memory During Request

In another terminal:
```bash
watch -n 1 nvidia-smi
```

You should see VRAM usage spike when processing, but it should have room to work.

## Expected Results

After fix, `nvidia-smi` should show:
- At rest: ~13-14 GB used (80-85%)
- During inference: ~15-15.5 GB used (leaving some headroom)

## If Still Issues

The Gemma 3n E2B model might just be too large for 16GB with vision.

**Alternative: Use a quantized or smaller model that's proven to work on 16GB:**

```bash
# Llama-3.2-11B-Vision (fits well on 16GB)
vllm serve meta-llama/Llama-3.2-11B-Vision-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 2048 \
  --limit-mm-per-prompt image=1
```

This model is specifically designed for vision and optimized for consumer GPUs.


