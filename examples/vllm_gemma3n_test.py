import sys
from vllm import LLM, EngineArgs

# --- Конфигурация ---
model_id = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"

engine_args = EngineArgs(
    model=model_id,
    tokenizer=model_id,
    quantization="bitsandbytes",     # Как в вашем логе
    trust_remote_code=True,       # Важно для MM/кастомных моделей
    dtype="bfloat16",             # Важно для bnb-4bit

    # --- Лимиты памяти (как в CLI) ---
    max_model_len=1024,
    gpu_memory_utilization=0.85,

    # --- КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ (те, что CLI не принял) ---
    max_num_mm_tokens=2048,       # Ограничиваем бюджет токенов для vision
    mm_profile_num_images=4         # Уменьшаем кол-во "dummy" образов для профилирования
)
# ---------------------

print(f"Попытка загрузить модель: {model_id}")
print("Используются следующие лимиты для Vision-энкодера:")
print(f"  max_num_mm_tokens = {engine_args.max_num_mm_tokens}")
print(f"  mm_profile_num_images = {engine_args.mm_profile_num_images}")

try:
    # Это тот самый шаг, который падал с OOM в вашем логе
    llm = LLM(engine_args=engine_args)

    print("\n✅ ✅ ✅ УСПЕХ! Модель загружена в VRAM.")
    print("OOM при профилировании Vision-энкодера предотвращен.")

    # (Опционально) Можете раскомментировать для запуска простого теста
    # from vllm import SamplingParams
    # prompts = ["<image>\nWhat is this image?", "What is 2+2?"]
    # sampling_params = SamplingParams(max_tokens=128, temperature=0.7)
    # print("\nЗапуск тестовой генерации...")
    # outputs = llm.generate(prompts, sampling_params)
    # print(outputs[1].outputs[0].text) # Печатаем ответ на "What is 2+2?"

except Exception as e:
    print(f"\n❌ ❌ ❌ ОШИБКА ЗАГРУЗКИ:\n{e}")
    sys.exit(1)
