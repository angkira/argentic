# Argentic Web Server

FastAPI-based backend for the Argentic visual agent builder.

## Installation

### Using uv (Recommended)

```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the Server

### Development

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Configuration

Set environment variables or create a `.env` file:

```env
ARGENTIC_WEB_DEBUG=false
ARGENTIC_WEB_HOST=0.0.0.0
ARGENTIC_WEB_PORT=8000
ARGENTIC_WEB_CORS_ORIGINS=http://localhost:4200,http://localhost:3000
```

## API Documentation

Once the server is running:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Project Structure

```
server/
├── app/
│   ├── models/          # Pydantic data models
│   │   ├── agent.py
│   │   ├── supervisor.py
│   │   ├── workflow.py
│   │   ├── config.py
│   │   └── ...
│   ├── routes/          # API route handlers
│   │   ├── agents.py
│   │   ├── supervisors.py
│   │   ├── workflows.py
│   │   └── config.py
│   ├── services/        # Business logic layer
│   │   ├── agent_service.py
│   │   ├── supervisor_service.py
│   │   ├── workflow_service.py
│   │   └── runtime_service.py
│   ├── core/            # Core utilities
│   │   ├── config.py
│   │   └── dependencies.py
│   └── main.py          # FastAPI application
├── scripts/
│   ├── generate_typescript_models.py
│   └── update_models.sh
├── tests/
├── pyproject.toml
└── main.py              # Entry point
```

## TypeScript Model Generation

Generate TypeScript interfaces from Pydantic models:

```bash
python scripts/generate_typescript_models.py
```

Or use the npm script from the client directory:

```bash
cd ../client
npm run generate-models
```

This reads the OpenAPI schema and generates TypeScript interfaces in `../client/src/app/models/generated.ts`.

## Development Tools

### Code Formatting

```bash
black app/ --line-length 100
```

### Linting

```bash
ruff check app/
```

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=app --cov-report=html
```

## Adding New Endpoints

1. Create/update Pydantic models in `app/models/`
2. Create route handlers in `app/routes/`
3. Add business logic in `app/services/`
4. Register router in `app/main.py`
5. Regenerate TypeScript models

Example:

```python
# app/models/my_model.py
from pydantic import BaseModel

class MyModel(BaseModel):
    name: str
    value: int

# app/routes/my_route.py
from fastapi import APIRouter
from app.models import MyModel

router = APIRouter(prefix="/api/my-endpoint", tags=["my-endpoint"])

@router.post("")
async def create_item(item: MyModel):
    return item

# app/main.py
from app.routes import my_route
app.include_router(my_route.router)
```

## Dependencies

Core dependencies:

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `argentic` - Argentic library

Dev dependencies:

- `pytest` - Testing
- `httpx` - Test client
- `black` - Code formatting
- `ruff` - Linting

## Environment Variables

- `ARGENTIC_WEB_APP_NAME` - Application name (default: "Argentic Web Server")
- `ARGENTIC_WEB_APP_VERSION` - Version (default: "0.1.0")
- `ARGENTIC_WEB_DEBUG` - Debug mode (default: false)
- `ARGENTIC_WEB_HOST` - Server host (default: "0.0.0.0")
- `ARGENTIC_WEB_PORT` - Server port (default: 8000)
- `ARGENTIC_WEB_CORS_ORIGINS` - Allowed CORS origins (comma-separated)

## License

MIT
