# Automatic TypeScript Model Generation

This project automatically generates TypeScript interfaces from Python Pydantic models to ensure type safety between backend and frontend.

## How It Works

1. **Python Models** (`web/server/app/models/`) - Define your Pydantic models
2. **FastAPI OpenAPI** - FastAPI automatically generates OpenAPI schema from Pydantic models
3. **Generation Script** (`web/server/scripts/generate_typescript_models.py`) - Reads OpenAPI schema and converts to TypeScript
4. **TypeScript Output** (`web/client/src/app/models/generated.ts`) - Auto-generated interfaces

## Usage

### Generate Models

From the client directory:
```bash
cd web/client
npm run generate-models
```

Or directly from server:
```bash
cd web/server
uv run python scripts/generate_typescript_models.py
```

Or use the shell script:
```bash
cd web/server
chmod +x generate_models.sh
./generate_models.sh
```

### When to Regenerate

Run model generation whenever you:
- ✅ Add new Pydantic models
- ✅ Modify existing model fields
- ✅ Change field types or constraints
- ✅ Update field descriptions
- ✅ Add/remove optional fields

### What Gets Generated

#### Python Input (Pydantic):
```python
# web/server/app/models/agent.py
class AgentConfig(BaseModel):
    """Agent configuration"""
    role: str = Field(..., description="Agent role/identifier", min_length=1)
    description: str = Field(..., description="Agent description")
    system_prompt: Optional[str] = None
    expected_output_format: Literal["json", "text", "code"] = "json"
    enable_dialogue_logging: bool = False
    max_consecutive_tool_calls: int = 3
```

#### TypeScript Output (Generated):
```typescript
// web/client/src/app/models/generated.ts
export interface AgentConfig {
  /** Agent role/identifier */
  role: string;
  /** Agent description */
  description: string;
  system_prompt?: string | null;
  expected_output_format: 'json' | 'text' | 'code';
  enable_dialogue_logging: boolean;
  max_consecutive_tool_calls: number;
}
```

## Type Conversions

| Python Type | TypeScript Type |
|------------|-----------------|
| `str` | `string` |
| `int`, `float` | `number` |
| `bool` | `boolean` |
| `Optional[T]` | `T \| null` |
| `List[T]` | `T[]` |
| `Dict[str, T]` | `Record<string, T>` |
| `Literal["a", "b"]` | `'a' \| 'b'` |
| `Union[A, B]` | `A \| B` |
| Nested models | Referenced interfaces |

## Features

### ✅ Readonly Properties
Response models (ending with `Response` or `Info`) automatically get `readonly` properties:
```typescript
export interface AgentResponse {
  readonly id: string;
  readonly role: string;
  readonly status: 'running' | 'stopped';
}
```

### ✅ JSDoc Comments
Field descriptions from Pydantic become JSDoc comments:
```typescript
export interface AgentCreate {
  /** Agent role/identifier */
  role: string;
}
```

### ✅ Optional vs Required
Pydantic's required/optional fields are preserved:
```python
# Python
system_prompt: Optional[str] = None  # Optional in Pydantic

# TypeScript
system_prompt?: string | null;  // Optional in TypeScript
```

### ✅ Enum Types
Literal types are converted to union types:
```python
# Python
output_format: Literal["json", "text", "code"]

# TypeScript
output_format: 'json' | 'text' | 'code';
```

## Project Structure

```
web/
├── server/
│   ├── app/
│   │   └── models/          # Python Pydantic models (source)
│   │       ├── agent.py
│   │       ├── supervisor.py
│   │       └── workflow.py
│   └── scripts/
│       └── generate_typescript_models.py  # Generation script
│
└── client/
    └── src/app/models/
        ├── generated.ts      # Auto-generated (don't edit!)
        ├── message-bus.ts    # Frontend-only models
        └── index.ts          # Export all models
```

## Hybrid Approach (Recommended)

Use both generated and manual models:

**For Backend API contracts:** Use generated models
```typescript
import { AgentCreate, AgentResponse } from './models/generated';
```

**For Frontend-only types:** Create manual models
```typescript
// models/message-bus.ts - Not synced with backend
export interface MessageBusFilter {
  agentId?: string;
  topic?: string;
}
```

**Export both:**
```typescript
// models/index.ts
export * from './generated';     // Backend models
export * from './message-bus';   // Frontend models
```

## Important Notes

### ⚠️ Don't Edit Generated Files
The `generated.ts` file is completely auto-generated. Any manual edits will be lost on next generation.

### ⚠️ Dependencies Required
The generation script requires:
- Python 3.11+
- uv package manager
- FastAPI and dependencies installed

### ⚠️ Build Required
The backend must be buildable for model generation to work. Make sure:
```bash
cd web/server
uv sync  # Install dependencies first
```

## Troubleshooting

### Error: Module not found
```bash
# Make sure dependencies are installed
cd web/server
uv sync
```

### Error: Unable to determine which files to ship
This is fixed - the `pyproject.toml` now includes:
```toml
[tool.hatch.build.targets.wheel]
packages = ["app"]
```

### Generation runs but produces empty file
Check that:
1. Your Pydantic models are properly defined
2. Models are imported in routes
3. Routes are registered with FastAPI app

## Best Practices

### ✅ DO
- Run generation after changing backend models
- Commit `generated.ts` to git
- Use generated types in API calls
- Keep frontend-only models separate
- Add generation to your CI/CD pipeline

### ❌ DON'T
- Edit `generated.ts` manually
- Mix backend and frontend types in same file
- Skip generation after backend changes
- Ignore TypeScript errors from generated types

## CI/CD Integration

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit
cd web/client && npm run generate-models
git add web/client/src/app/models/generated.ts
```

### GitHub Actions
```yaml
- name: Generate TypeScript Models
  run: |
    cd web/client
    npm run generate-models
```

## Example Usage

### Creating an Agent
```typescript
import { AgentCreate, AgentResponse } from './models';
import { ApiService } from './services/api.service';

// Type-safe API call
const newAgent: AgentCreate = {
  role: 'research_agent',
  description: 'Research assistant',
  expected_output_format: 'json',  // Auto-complete works!
  enable_dialogue_logging: true,
  max_consecutive_tool_calls: 5
};

// TypeScript ensures correct types
this.apiService.createAgent(newAgent)
  .subscribe((response: AgentResponse) => {
    console.log(response.id);     // readonly property
    console.log(response.status); // 'running' | 'stopped' | 'error'
  });
```

### Type Safety in Components
```typescript
// Declarative stream with generated types
readonly agents$: Observable<AgentResponse[]> =
  this.apiService.getAgents();

// Async pipe with full type safety
@if (agents$ | async; as agents) {
  @for (agent of agents; track agent.id) {
    <div>{{ agent.role }}</div>
  }
}
```

## Summary

✨ **Automatic type safety** between Python and TypeScript
✨ **No manual synchronization** needed
✨ **Always up-to-date** with backend models
✨ **Full IDE support** with autocomplete and type checking
✨ **Reduced bugs** from type mismatches

Run `npm run generate-models` after changing backend models, and enjoy type-safe development!
