# Тест Gemma JAX с gcloud auth

Ты залогинен в gcloud, теперь можно попробовать JAX provider.

## Как это работает

1. `gemma` library пытается загрузить из `gs://gemma-data/checkpoints/gemma3n-e4b-it`
2. Использует gcloud credentials для доступа к GCS
3. Автоматически скачивает модель (~10GB) в `~/.cache/` при первом запуске

## Тест

```bash
cd /home/angkira/Project/software/argentic

# Проверь gcloud auth работает
gcloud auth application-default print-access-token

# Запусти агента (автоматически скачает модель при первом запуске)
uv run python examples/visual_agent_gemma_jax.py
```

## Ожидаемое поведение

**Первый запуск:**
```
2. Loading Gemma 3n E4B model (JAX)...
   (Loading from local checkpoint - ~10GB RAM needed)
2025-10-22 XX:XX:XX [INFO] gemma_jax: Loading Gemma model with JAX...
2025-10-22 XX:XX:XX [INFO] gemma_jax: Auto-downloading checkpoint: gs://gemma-data/checkpoints/gemma3n-e4b-it
# Скачивание ~10GB (5-15 минут)
2025-10-22 XX:XX:XX [INFO] gemma_jax: ✓ Gemma model loaded
   ✓ Model ready
```

**Последующие запуски:**
Модель в кеше, загрузка быстрая (~10 секунд).

## Если ошибка

### "Could not find credentials file"

```bash
# Переавторизуйся
gcloud auth application-default login
```

### "AttributeError: item_metadata"

Баг в orbax. Используй vLLM вместо JAX.

### "FileNotFoundError: _METADATA"

Это означает что Kaggle checkpoint не подходит. Используй auto-download (без `gemma_checkpoint_path` в конфиге).

## Вывод

После теста:
- **Если работает** - можешь использовать JAX для экспериментов
- **Если не работает** - забей, используй vLLM (он надежнее и быстрее)

JAX provider оставлен в коде **на всякий случай**, но **vLLM - основной вариант**.

