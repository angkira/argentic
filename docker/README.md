# Argentic Docker Deployment

Complete Docker setup for running Argentic with various LLM backends.

## Quick Start

### 1. Prerequisites

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### 2. Setup Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with your credentials
nano .env
```

### 3. Choose Your Backend

#### Option A: Ollama (Easiest, Text-Only)

```bash
# Start Ollama
docker-compose up -d ollama

# Pull Gemma 3n
docker exec ollama ollama pull gemma3n:4b

# Test
docker exec ollama ollama run gemma3n:4b "Hello!"
```

#### Option B: Transformers + PaliGemma (Working Multimodal)

```bash
# Build
docker-compose build argentic-transformers

# Run (will auto-download model ~3GB)
docker-compose up argentic-transformers

# Logs
docker-compose logs -f argentic-transformers
```

#### Option C: JAX + Gemma (Official, Full Multimodal)

```bash
# Set Kaggle credentials in .env first!

# Build
docker-compose build argentic-gemma-jax

# Run (will download model ~4GB on first run)
docker-compose up argentic-gemma-jax

# Logs
docker-compose logs -f argentic-gemma-jax
```

## Architecture

```
┌─────────────────────────────────────┐
│         docker-compose.yml          │
├─────────────────────────────────────┤
│                                     │
│  ┌────────────┐  ┌──────────────┐ │
│  │ Mosquitto  │  │   Ollama     │ │
│  │ (MQTT)     │  │ (optional)   │ │
│  └────────────┘  └──────────────┘ │
│         │                          │
│         │                          │
│  ┌──────────────────────────────┐ │
│  │  Argentic with:              │ │
│  │  - JAX Gemma (multimodal)    │ │
│  │  OR                          │ │
│  │  - Transformers (PaliGemma)  │ │
│  └──────────────────────────────┘ │
│                                     │
└─────────────────────────────────────┘
```

## Services

### mosquitto
- **Image:** `eclipse-mosquitto:2.0`
- **Ports:** 1883 (MQTT), 9001 (WebSocket)
- **Purpose:** Message broker for Argentic agents

### ollama
- **Image:** `ollama/ollama:latest`
- **Ports:** 11434 (REST API)
- **GPU:** Required
- **Models:** Text-only Gemma 3n

### argentic-transformers
- **Base:** `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- **GPU:** Required
- **Models:** PaliGemma (multimodal, working)
- **VRAM:** ~8GB

### argentic-gemma-jax
- **Base:** `nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04`
- **GPU:** Required
- **Models:** Gemma 3n E4B (official, multimodal)
- **VRAM:** ~8GB

## Commands

```bash
# Build all services
docker-compose build

# Start specific service
docker-compose up argentic-gemma-jax

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all
docker-compose down

# Clean up (including volumes)
docker-compose down -v

# Rebuild from scratch
docker-compose build --no-cache

# Shell into container
docker-compose exec argentic-gemma-jax bash

# Run one-off command
docker-compose run --rm argentic-gemma-jax python -c "from gemma import gm; print(gm.__version__)"
```

## Resource Requirements

| Service | GPU | VRAM | RAM | Disk |
|---------|-----|------|-----|------|
| **mosquitto** | No | - | 100MB | 50MB |
| **ollama** | Yes | 6GB | 4GB | 5GB |
| **transformers** | Yes | 8GB | 8GB | 15GB |
| **gemma-jax** | Yes | 8GB | 8GB | 20GB |

## Configuration

### docker-compose.yml
Main orchestration file. Edit to:
- Change GPU allocation
- Adjust memory limits
- Add/remove services
- Modify port mappings

### .env
Environment variables:
- `KAGGLE_USERNAME` / `KAGGLE_KEY` - For Gemma model downloads
- `HF_TOKEN` - HuggingFace token (optional)
- `HF_MODEL_ID` - Model to use with Transformers

### Dockerfiles
- `Dockerfile.transformers` - PyTorch-based setup
- `Dockerfile.gemma-jax` - JAX-based setup

## Troubleshooting

### GPU not detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# If fails, check daemon config
cat /etc/docker/daemon.json

# Should contain:
{
  "runtimes": {
    "nvidia": {
      "path": "/usr/bin/nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

# Restart Docker
sudo systemctl restart docker
```

### Out of memory

```bash
# Check current usage
nvidia-smi

# Reduce resources in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G  # Reduce this
```

### Model download fails

```bash
# For JAX Gemma: Check Kaggle credentials
echo $KAGGLE_USERNAME
echo $KAGGLE_KEY

# For Transformers: Pre-download model
python -c "
from transformers import AutoModel
AutoModel.from_pretrained('google/paligemma-3b-mix-224')
"

# Then mount as volume
volumes:
  - ~/.cache/huggingface:/root/.cache/huggingface
```

### Container crashes

```bash
# Check logs
docker-compose logs argentic-gemma-jax

# Check container status
docker-compose ps

# Inspect container
docker inspect argentic-gemma-jax
```

## Development

For active development, mount source code:

```yaml
# docker-compose.dev.yml
services:
  argentic-gemma-jax:
    volumes:
      - .:/app  # Mount entire codebase
    command: python examples/visual_agent_gemma_jax.py
```

Run:
```bash
docker-compose -f docker-compose.dev.yml up
```

## Production

For production deployment:

1. Use multi-stage builds (see `Dockerfile.gemma-jax`)
2. Enable health checks
3. Set restart policies
4. Add proper logging
5. Secure MQTT with authentication
6. Use secrets for credentials

```yaml
# docker-compose.prod.yml
services:
  argentic-gemma-jax:
    restart: always
    secrets:
      - kaggle_username
      - kaggle_key
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Resources

- **Ollama:** https://ollama.com/
- **JAX:** https://jax.readthedocs.io/
- **Gemma:** https://github.com/google-deepmind/gemma
- **HuggingFace:** https://huggingface.co/
- **NVIDIA Container Toolkit:** https://github.com/NVIDIA/nvidia-container-toolkit

