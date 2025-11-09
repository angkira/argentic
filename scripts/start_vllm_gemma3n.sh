#!/bin/bash
# Start vLLM server for Gemma 3N vision model
#
# Usage: ./scripts/start_vllm_gemma3n.sh [model_path]
#
# Default model path: ~/models/gemma-3-4b-4bit/gemma-3n-E4B-it-quantized

MODEL_PATH="${1:-$HOME/models/gemma-3-4b-4bit/gemma-3n-E4B-it-quantized}"
PORT="${2:-8000}"

echo "Starting vLLM server..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo ""
echo "IMPORTANT: This model requires --trust-remote-code flag!"
echo ""

vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --max-num-seqs 2

# Note: The model documentation specifies:
# - vLLM >= 0.10.0 required
# - trust_remote_code=True is REQUIRED for custom model components
# - Recommended: max_model_len=4096, max_num_seqs=2
