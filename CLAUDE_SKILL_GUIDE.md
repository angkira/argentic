# Claude Skill для Argentic - Руководство по использованию

## Что это?

Claude Skill - это специальный формат документации от Anthropic, который позволяет Claude загружать экспертные знания о фреймворках и библиотеках. 

Для Argentic создан skill, который даёт Claude полное понимание:
- Архитектуры фреймворка
- Паттернов разработки (single agent, custom tools, multi-agent)
- API всех компонентов
- Best practices и troubleshooting
- Рабочих примеров кода

## Файлы

### 1. argentic-skill.zip (~12 KB)
Готовый к загрузке в Claude архив, содержащий:

- **SKILL.md** - Основной файл с экспертными знаниями (~900 строк)
  - Framework overview
  - 3 полных паттерна (single agent, custom tool, multi-agent)
  - Configuration examples
  - Complete API reference
  - Best practices
  - Troubleshooting guide

- **README.md** - Документация по использованию skill
  
- **examples/** - Рабочие примеры кода:
  - `single_agent.py` - Базовый агент
  - `custom_tool.py` - Создание инструмента
  - `multi_agent.py` - Мульти-агентная система

### 2. argentic-skill/ (папка)
Исходники skill для редактирования и обновления.

## Как установить в Claude

### Вариант 1: Claude Desktop (macOS/Windows)

1. **Откройте настройки:**
   - macOS: `Claude → Preferences → Features`
   - Windows: `Settings → Features`

2. **Добавьте skill:**
   - Найдите раздел "Skills" или "Custom Skills"
   - Нажмите "Add Skill" или "Upload Skill"
   - Выберите файл `argentic-skill.zip`
   - Подтвердите загрузку

3. **Проверьте установку:**
   - Skill должен появиться в списке доступных
   - Статус: "Active" или "Enabled"

### Вариант 2: Claude Code (VS Code Extension)

1. **Откройте Command Palette:**
   - `Cmd/Ctrl + Shift + P`

2. **Найдите команду:**
   - Введите "Claude: Manage Skills"
   - Или "Claude: Add Skill"

3. **Загрузите skill:**
   - Выберите `argentic-skill.zip`
   - Дождитесь подтверждения

### Вариант 3: Claude Web (если поддерживается)

На момент написания (октябрь 2025) Skills могут быть доступны только в Desktop/Code версиях. Проверьте документацию Claude для актуальной информации.

## Как это работает

### Автоматическая активация

Claude автоматически определяет, когда использовать skill на основе:

1. **Зависимостей проекта:**
   ```python
   # requirements.txt
   argentic>=0.11.0
   
   # pyproject.toml
   dependencies = ["argentic>=0.11.0"]
   ```

2. **Импортов в коде:**
   ```python
   from argentic import Agent, Messager
   from argentic.core.tools import BaseTool
   ```

3. **Явных запросов:**
   - "Using Argentic framework, create..."
   - "Build an Argentic agent with..."
   - "Help me with Argentic multi-agent system"

### Примеры использования

**Пример 1: Создание агента**
```
User: Create a simple AI agent using Argentic with Google Gemini
Claude: [Использует skill] Sure! Here's a complete implementation...
```

**Пример 2: Custom tool**
```
User: I need to create a weather tool for my Argentic agent
Claude: [Использует skill] I'll create a weather tool following Argentic patterns...
```

**Пример 3: Multi-agent**
```
User: Build a multi-agent system with researcher and analyst
Claude: [Использует skill] I'll implement a supervisor-based multi-agent system...
```

**Пример 4: Debugging**
```
User: My tool isn't registering with the agent
Claude: [Использует skill] This is a common issue. Here are the steps to fix it...
```

## Преимущества

### ✅ Мгновенная экспертиза
Claude сразу знает все паттерны Argentic без дополнительного контекста.

### ✅ Экономия контекста
Skill не занимает место в вашем context window - загружается динамически.

### ✅ Правильные паттерны
Claude автоматически использует best practices из skill.

### ✅ Актуальные примеры
Все примеры работают с текущей версией Argentic (0.11.x).

### ✅ Troubleshooting
Claude знает все типичные проблемы и их решения.

## Сравнение с другими форматами

| Формат | Claude Skill | .cursorrules | QUICKREF.md |
|--------|-------------|--------------|-------------|
| Работает в Claude Desktop/Code | ✅ | ❌ | ❌ |
| Работает в Cursor | ❌ | ✅ | ✅ |
| Экономит context window | ✅ | ❌ | ❌ |
| Автоматическая активация | ✅ | ✅ (Cursor) | ❌ |
| Version control | ⚠️ (нужен re-upload) | ✅ | ✅ |
| Включает бинарные файлы | ✅ | ❌ | ❌ |

**Рекомендация:** Используйте все три формата для максимальной совместимости:
- **Claude Skill** - для Claude Desktop/Code
- **.cursorrules** - для Cursor AI
- **QUICKREF.md** - для ручного reference

## Обновление Skill

Когда Argentic обновляется:

1. **Отредактируйте файлы:**
   ```bash
   cd argentic-skill/
   # Редактируйте SKILL.md
   nano SKILL.md
   ```

2. **Обновите примеры:**
   ```bash
   # Редактируйте examples/
   nano examples/single_agent.py
   ```

3. **Пересоздайте ZIP:**
   ```bash
   cd ..
   python3 << 'EOF'
   import zipfile
   from pathlib import Path
   
   with zipfile.ZipFile('argentic-skill.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
       for file in Path('argentic-skill').rglob('*'):
           if file.is_file():
               zipf.write(file, file.relative_to('argentic-skill').parent)
   EOF
   ```

4. **Перезагрузите в Claude:**
   - Удалите старый skill в настройках
   - Загрузите новый `argentic-skill.zip`

## Проверка работы

### Тест 1: Проверка активации

1. Создайте новый чат с Claude
2. Напишите: "I'm using Argentic framework"
3. Спросите: "How do I create a single agent?"
4. Claude должен дать точный ответ с кодом

### Тест 2: Проверка примеров

1. Спросите: "Show me a complete example of custom tool in Argentic"
2. Claude должен предоставить рабочий код с Pydantic схемой

### Тест 3: Проверка troubleshooting

1. Спросите: "My Argentic tool isn't registering, how to fix?"
2. Claude должен дать конкретные шаги решения

## Устранение проблем

### Skill не загружается

**Причина:** Неправильная структура ZIP

**Решение:**
- Убедитесь, что SKILL.md в корне архива
- Проверьте YAML front matter в SKILL.md
- Пересоздайте ZIP согласно инструкции выше

### Claude не использует skill

**Причина:** Неясный контекст проекта

**Решение:**
- Явно упомяните "Argentic" в запросе
- Добавьте import argentic в код
- Укажите в project settings

### Устаревшая информация

**Причина:** Skill не обновлён

**Решение:**
- Проверьте версию Argentic в skill
- Обновите skill согласно инструкции выше
- Перезагрузите в Claude

## FAQ

**Q: Нужен ли skill для каждого проекта?**  
A: Нет, skill загружается один раз и работает для всех проектов.

**Q: Можно ли использовать несколько skills одновременно?**  
A: Да, Claude может использовать несколько skills параллельно.

**Q: Skill занимает место в моём плане?**  
A: Нет, skills не учитываются в лимитах использования.

**Q: Можно ли поделиться skill с командой?**  
A: Да, просто передайте файл `argentic-skill.zip`.

**Q: Как часто нужно обновлять skill?**  
A: При выходе новых версий Argentic с breaking changes.

## Дополнительные ресурсы

- **Argentic Documentation**: См. `docs/` директорию
- **Examples**: См. `examples/` директорию  
- **API Reference**: `ARGENTIC_QUICKREF.md`
- **Cursor Rules**: `.cursorrules`
- **GitHub**: https://github.com/angkira/argentic

## Поддержка

Если skill не работает или нужна помощь:

1. Проверьте версию Claude Desktop/Code
2. Убедитесь, что Skills поддерживаются в вашей версии
3. Проверьте логи Claude (если доступны)
4. Попробуйте пересоздать и перезагрузить skill

---

**Версия skill:** 1.0  
**Для Argentic:** 0.11.x  
**Создано:** Октябрь 2025  
**Формат:** Claude Skills (Anthropic)

