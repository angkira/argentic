# Запуск Gemma 3n

## vLLM Provider (рекомендуется) ✅

### 1. Запусти vLLM сервер

```bash
# Базовый запуск
vllm serve google/gemma-3n-E4B-it --port 8000

# С GPU оптимизациями
vllm serve google/gemma-3n-E4B-it \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096
```

### 2. Запусти агента

```bash
cd /home/angkira/Project/software/argentic
source .env

# Простой тест
uv run python examples/test_vllm_simple.py

# Visual agent
uv run python examples/visual_agent_vllm.py
```

## JAX Provider (экспериментальный) ⚠️

**Статус:** Работает только с gcloud auth, багованный.

### Setup с gcloud

```bash
# 1. Авторизуйся в gcloud (один раз)
gcloud auth application-default login

# 2. Запусти агента
cd /home/angkira/Project/software/argentic
uv run python examples/visual_agent_gemma_jax.py
```

**Как работает:**
- gemma library скачивает модель из `gs://gemma-data/checkpoints/`
- Требуется gcloud auth (Google Cloud Storage)
- Автоматическая загрузка при первом запуске (~10GB)

**Проблемы:**
- ❌ Кривая библиотека с багами orbax
- ❌ Требует gcloud setup
- ❌ Медленнее vLLM
- ❌ Нет PagedAttention и прочих оптимизаций

### Если не работает

```bash
# Проверь gcloud auth
gcloud auth application-default print-access-token

# Или используй vLLM вместо JAX
```

---

## Сравнение

| Параметр       | vLLM Provider           | JAX Provider              |
|----------------|-------------------------|---------------------------|
| Setup          | ✅ Простой              | ❌ Сложный (gcloud нужен) |
| Model download | ✅ Auto от HuggingFace  | ⚠️ Auto от GCS (gcloud)   |
| Performance    | ✅ Высокий (PagedAttn)  | ⚠️ Средний                |
| Memory         | ✅ Эффективная          | ⚠️ Стандартная            |
| API            | ✅ OpenAI-compatible    | Custom                    |
| Stability      | ✅ Production-ready     | ❌ Багованный             |

**Рекомендация:** Используй **vLLM** для production. JAX только для экспериментов.

Подробнее: `VLLM_SETUP.md`
