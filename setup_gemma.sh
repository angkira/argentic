#!/bin/bash
# Setup script for Gemma 3n Visual Agent

set -e

echo "=========================================="
echo "Gemma 3n Visual Agent Setup"
echo "=========================================="
echo ""

# Check Python
echo "1. Checking Python..."
python --version || { echo "Python not found!"; exit 1; }
echo "   ✓ Python OK"
echo ""

# Install dependencies
echo "2. Installing dependencies..."
pip install transformers accelerate torch torchvision timm huggingface-hub pillow || {
    echo "   Failed to install packages!"
    exit 1
}
echo "   ✓ Dependencies installed"
echo ""

# Download model
echo "3. Downloading Gemma 3n E4B model..."
echo "   This will take 10-30 minutes (8-10 GB)"
echo ""
python -c "
from huggingface_hub import snapshot_download
import os

model_id = 'google/gemma-3n-E4B-it'
print(f'Downloading {model_id}...')
print('Files will be cached in ~/.cache/huggingface/')
print('')

try:
    path = snapshot_download(
        repo_id=model_id,
        ignore_patterns=['*.md', '*.txt', 'LICENSE'],
    )
    print(f'✓ Model downloaded to: {path}')
except Exception as e:
    print(f'Error: {e}')
    print('')
    print('If you see authentication error, run:')
    print('huggingface-cli login')
    exit(1)
"
echo ""

# Start MQTT
echo "4. Starting MQTT broker..."
if command -v docker &> /dev/null; then
    docker ps | grep mosquitto > /dev/null || {
        echo "   Starting mosquitto container..."
        docker run -d -p 1883:1883 --name mosquitto-argentic eclipse-mosquitto:2.0
    }
    echo "   ✓ MQTT broker running"
else
    echo "   ⚠ Docker not found - please start MQTT manually"
    echo "   Install: docker run -d -p 1883:1883 eclipse-mosquitto:2.0"
fi
echo ""

echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To run Gemma agent:"
echo "  python examples/visual_agent_gemma_webrtc.py"
echo ""
echo "To test with Gemini API (no model needed):"
echo "  python examples/visual_agent_gemini_test.py"
echo ""

