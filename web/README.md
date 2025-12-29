# Argentic Web - Visual Agent Builder

A web-based visual interface for building and managing Argentic AI agents, supervisors, and workflows. Built with FastAPI (backend) and Angular 20+ (frontend).

## Features

- **Visual Workflow Builder**: Drag-and-drop interface for creating agent workflows (n8n-style)
- **Agent Management**: Create, configure, and monitor AI agents
- **Supervisor Management**: Build multi-agent systems with supervisors
- **Configuration UI**: Easy setup for LLM providers and messaging backends
- **Real-time Monitoring**: Track agent status and execution
- **TypeScript Model Generation**: Auto-generate frontend models from Python Pydantic models

## Architecture

```
web/
├── server/          # FastAPI backend
│   ├── app/         # Application code
│   │   ├── models/  # Pydantic models
│   │   ├── routes/  # API endpoints
│   │   ├── services/# Business logic
│   │   └── core/    # Core utilities
│   ├── scripts/     # Utility scripts
│   └── tests/       # Backend tests
│
└── client/          # Angular 20 frontend
    ├── src/
    │   └── app/
    │       ├── components/  # UI components
    │       ├── models/      # TypeScript interfaces
    │       └── services/    # API services
    └── angular.json
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- uv (Python package manager)
- MQTT broker (for messaging, e.g., Mosquitto)

### 1. Start the Backend

```bash
cd server

# Install dependencies with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Run the server
python main.py
```

The API will be available at `http://localhost:8000`

### 2. Start the Frontend

```bash
cd client

# Install dependencies
npm install

# Start development server
npm start
```

The UI will be available at `http://localhost:4200`

## Development

### Generate TypeScript Models from Python

The project includes automatic TypeScript interface generation from Pydantic models:

```bash
# From the server directory
python scripts/generate_typescript_models.py

# Or from the client directory
npm run generate-models
```

This creates `client/src/app/models/generated.ts` with all API models.

### API Documentation

FastAPI provides interactive API documentation:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Project Structure Details

#### Backend (FastAPI)

- **Models** (`app/models/`): Pydantic models for API validation
  - `agent.py`: Agent configuration models
  - `supervisor.py`: Supervisor configuration models
  - `workflow.py`: Workflow graph models
  - `config.py`: LLM and messaging configuration models

- **Routes** (`app/routes/`): API endpoints
  - `agents.py`: Agent CRUD operations
  - `supervisors.py`: Supervisor CRUD operations
  - `workflows.py`: Workflow CRUD operations
  - `config.py`: Configuration endpoints

- **Services** (`app/services/`): Business logic
  - `agent_service.py`: Agent management
  - `supervisor_service.py`: Supervisor management
  - `workflow_service.py`: Workflow management
  - `runtime_service.py`: Agent execution and monitoring

#### Frontend (Angular 20)

- **Components** (`src/app/components/`):
  - `dashboard/`: Overview dashboard
  - `agents/`: Agent management UI
  - `supervisors/`: Supervisor management UI
  - `workflows/`: Workflow list
  - `workflow-builder/`: Visual workflow editor
  - `config/`: Configuration settings

- **Models** (`src/app/models/`): TypeScript interfaces matching backend
- **Services** (`src/app/services/`): API communication layer

### Angular 20+ Features Used

- **Standalone Components**: No NgModules required
- **New Control Flow**: `@if`, `@for`, `@switch` instead of `*ngIf`, `*ngFor`
- **Signals**: Reactive state management
- **Improved TypeScript**: TypeScript 5.7+ support

## Configuration

### Backend Configuration

Create a `.env` file in the `server` directory:

```env
# LLM Provider
GOOGLE_GEMINI_API_KEY=your_api_key_here

# Server Settings
ARGENTIC_WEB_DEBUG=false
ARGENTIC_WEB_HOST=0.0.0.0
ARGENTIC_WEB_PORT=8000

# CORS
ARGENTIC_WEB_CORS_ORIGINS=http://localhost:4200
```

### Frontend Configuration

The frontend uses a proxy configuration for API calls (see `client/proxy.conf.json`).

## API Endpoints

### Agents

- `GET /api/agents` - List all agents
- `POST /api/agents` - Create agent
- `GET /api/agents/{id}` - Get agent
- `PATCH /api/agents/{id}` - Update agent
- `DELETE /api/agents/{id}` - Delete agent
- `POST /api/agents/{id}/start` - Start agent
- `POST /api/agents/{id}/stop` - Stop agent

### Supervisors

- `GET /api/supervisors` - List all supervisors
- `POST /api/supervisors` - Create supervisor
- `GET /api/supervisors/{id}` - Get supervisor
- `PATCH /api/supervisors/{id}` - Update supervisor
- `DELETE /api/supervisors/{id}` - Delete supervisor
- `POST /api/supervisors/{id}/start` - Start supervisor
- `POST /api/supervisors/{id}/stop` - Stop supervisor

### Workflows

- `GET /api/workflows` - List all workflows
- `POST /api/workflows` - Create workflow
- `GET /api/workflows/{id}` - Get workflow
- `PATCH /api/workflows/{id}` - Update workflow
- `DELETE /api/workflows/{id}` - Delete workflow
- `POST /api/workflows/{id}/start` - Start workflow
- `POST /api/workflows/{id}/stop` - Stop workflow

### Configuration

- `GET /api/config/llm-providers` - List available LLM providers
- `GET /api/config/messaging-protocols` - List available messaging protocols
- `POST /api/config/llm-providers/validate` - Validate LLM configuration
- `POST /api/config/messaging/validate` - Validate messaging configuration

## Testing

### Backend Tests

```bash
cd server
pytest
```

### Frontend Tests

```bash
cd client
ng test
```

## Building for Production

### Backend

```bash
cd server
uv pip install -e .
```

Run with a production server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend

```bash
cd client
ng build --configuration production
```

Static files will be in `client/dist/argentic-web-client/`.

## Contributing

This is part of the Argentic project. See the main repository README for contribution guidelines.

## License

MIT License - see the main repository LICENSE file.

## Related Documentation

- [Argentic Main Documentation](../../README.md)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Angular Documentation](https://angular.dev/)
- [uv Package Manager](https://github.com/astral-sh/uv)
