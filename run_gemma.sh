#!/bin/bash
# Run Gemma 3n Visual Agent

set -e

echo "=========================================="
echo "Starting Gemma 3n Visual Agent"
echo "=========================================="
echo ""

# Check MQTT
echo "1. Checking MQTT broker..."
nc -zv localhost 1883 2>&1 | grep -q "succeeded" || {
    echo "   ⚠ MQTT not running on localhost:1883"
    echo "   Starting with Docker..."
    docker run -d -p 1883:1883 --name mosquitto-argentic eclipse-mosquitto:2.0 2>/dev/null || {
        echo "   Container already exists or Docker unavailable"
    }
    sleep 2
}
echo "   ✓ MQTT broker available"
echo ""

# Check GPU
echo "2. Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'   ✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('   ⚠ No GPU found - will use CPU (slow!)')
" || echo "   ⚠ Could not check GPU"
echo ""

# Check model
echo "3. Checking model..."
python -c "
from pathlib import Path
import os

# Check HF cache
hf_cache = Path.home() / '.cache' / 'huggingface' / 'hub'
model_dirs = list(hf_cache.glob('models--google--gemma-3n-E4B-it'))

if model_dirs:
    print(f'   ✓ Model found in cache: {model_dirs[0]}')
else:
    print('   ⚠ Model not found!')
    print('   Run: bash setup_gemma.sh')
    exit(1)
" || exit 1
echo ""

# Run
echo "4. Starting Visual Agent..."
echo ""
echo "=========================================="
echo ""

cd "$(dirname "$0")"
python examples/visual_agent_gemma_webrtc.py

echo ""
echo "=========================================="
echo "Agent stopped"
echo "=========================================="

