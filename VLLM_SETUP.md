# vLLM Provider Setup для Gemma 3n

vLLM - это высокопроизводительный inference engine с OpenAI-compatible API.

## Преимущества vLLM

- ✅ **PagedAttention** - эффективное использование GPU памяти
- ✅ **Continuous batching** - высокий throughput
- ✅ **OpenAI API** - drop-in replacement
- ✅ **Автозагрузка** - модели скачиваются из HuggingFace автоматически
- ✅ **Multimodal** - поддержка Gemma 3n (text + images + audio)

## Установка

```bash
# vLLM (GPU required)
pip install vllm

# Или через uv
uv pip install vllm
```

**Требования:**
- CUDA 11.8+ или 12.1+
- GPU с минимум 16GB VRAM для Gemma 3n E4B
- Python 3.8+

## Запуск vLLM сервера

### Gemma 3n E4B (multimodal)

```bash
# Базовый запуск
vllm serve google/gemma-3n-E4B-it --port 8000

# С GPU memory optimization
vllm serve google/gemma-3n-E4B-it \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096

# С tensor parallelism (для multi-GPU)
vllm serve google/gemma-3n-E4B-it \
  --port 8000 \
  --tensor-parallel-size 2
```

### Gemma 3n E2B (smaller, faster)

```bash
vllm serve google/gemma-3n-E2B-it --port 8000
```

### Проверка статуса

```bash
# Health check
curl http://localhost:8000/health

# Список моделей
curl http://localhost:8000/v1/models
```

## Использование в Argentic

### 1. Конфигурация

```yaml
# config.yaml
llm:
  provider: vllm
  vllm_base_url: http://localhost:8000/v1
  vllm_model_name: google/gemma-3n-E4B-it
  vllm_api_key: dummy  # vLLM не требует настоящий ключ
  temperature: 0.7
  max_tokens: 2048

messaging:
  broker_address: localhost
  port: 1883
```

### 2. Python код

```python
from argentic import Agent, Messager, LLMFactory
from argentic.core.tools import ToolManager

async def main():
    # Setup
    messager = Messager(broker_address="localhost", port=1883)
    await messager.connect()
    
    tool_manager = ToolManager(messager)
    await tool_manager.async_init()
    
    # vLLM provider
    config = {
        "llm": {
            "provider": "vllm",
            "vllm_base_url": "http://localhost:8000/v1",
            "vllm_model_name": "google/gemma-3n-E4B-it",
        }
    }
    llm = LLMFactory.create(config, messager)
    
    # Create agent
    agent = Agent(
        llm=llm,
        messager=messager,
        tool_manager=tool_manager,
        role="assistant",
        system_prompt="You are a helpful AI assistant.",
    )
    await agent.async_init()
    
    # Query
    response = await agent.query("What is the capital of France?")
    print(response)
```

### 3. Visual Agent

```python
from argentic.core.agent.visual_agent import VisualAgent

# Same setup as above, then:
visual_agent = VisualAgent(
    llm=llm,
    messager=messager,
    tool_manager=tool_manager,
    driver=webrtc_driver,  # Your WebRTC driver
    role="visual_assistant",
)
await visual_agent.async_init()

# Ask about image
response = await visual_agent.query_with_video("What's in this image?")
```

## Примеры запуска

### Text-only agent

```bash
# 1. Start vLLM server
vllm serve google/gemma-3n-E4B-it --port 8000

# 2. Run agent (в другом терминале)
source .env
uv run python examples/single_agent_vllm.py  # TODO: create this
```

### Visual agent

```bash
# 1. Start vLLM server with multimodal support
vllm serve google/gemma-3n-E4B-it --port 8000

# 2. Run visual agent
source .env
uv run python examples/visual_agent_vllm.py
```

## Производительность

### Gemma 3n E4B на разных GPU

| GPU           | VRAM | Batch Size | Throughput      |
|---------------|------|------------|-----------------|
| RTX 4090      | 24GB | 16         | ~50 tokens/sec  |
| A100 40GB     | 40GB | 32         | ~100 tokens/sec |
| A100 80GB     | 80GB | 64         | ~150 tokens/sec |

### Оптимизации

```bash
# KV cache quantization (FP8)
vllm serve google/gemma-3n-E4B-it \
  --port 8000 \
  --kv-cache-dtype fp8

# AWQ quantization (4-bit weights)
vllm serve google/gemma-3n-E4B-it \
  --port 8000 \
  --quantization awq

# Speculative decoding (faster inference)
vllm serve google/gemma-3n-E4B-it \
  --port 8000 \
  --speculative-model google/gemma-3n-E2B-it
```

## Troubleshooting

### OOM (Out of Memory)

```bash
# Уменьши max_model_len
vllm serve google/gemma-3n-E4B-it \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8

# Или используй меньшую модель
vllm serve google/gemma-3n-E2B-it --port 8000
```

### Connection refused

```bash
# Проверь что сервер запущен
curl http://localhost:8000/health

# Проверь порт
netstat -tulpn | grep 8000
```

### Slow first request

Первый запрос загружает модель в память (~30-60 секунд). Последующие запросы быстрые.

## Сравнение с JAX provider

| Параметр              | vLLM                  | JAX (gemma library)   |
|-----------------------|-----------------------|-----------------------|
| Setup                 | ✅ Простой (один pip) | ❌ Сложный (JAX+orbax)|
| Model loading         | ✅ Auto от HF         | ❌ Kaggle/GCS manual  |
| Memory efficiency     | ✅ PagedAttention     | ⚠️ Стандартная        |
| Throughput            | ✅ Высокий            | ⚠️ Средний            |
| API                   | ✅ OpenAI-compatible  | ❌ Custom             |
| Multi-GPU             | ✅ Поддержка          | ⚠️ Ограниченная       |
| Quantization          | ✅ FP8, AWQ, GPTQ     | ❌ Нет                |

**Вывод:** Для production используй **vLLM**, для research/experiments - JAX.

## Дополнительно

- [vLLM Documentation](https://docs.vllm.ai/)
- [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [Performance Benchmarks](https://blog.vllm.ai/2023/06/20/vllm.html)

