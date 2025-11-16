# Testing Guide for Argentic Web

This document describes the testing strategy and how to run tests for the Argentic Web application.

## Overview

The test suite includes:

- **Backend Tests** (Python/FastAPI)
  - Unit tests for models and services
  - Integration tests for API endpoints
- **Frontend Tests** (Angular/TypeScript)
  - Unit tests for components and services
  - E2E tests with Playwright

## Backend Testing

### Setup

```bash
cd web/server
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run only unit tests
pytest tests/unit -v

# Run only integration tests
pytest tests/integration -v

# Run specific test file
pytest tests/unit/test_models.py -v

# Run specific test
pytest tests/unit/test_models.py::TestAgentModels::test_agent_create_valid -v
```

### Test Structure

```
web/server/tests/
├── conftest.py              # Pytest configuration and fixtures
├── unit/                    # Unit tests
│   ├── test_models.py      # Pydantic model tests
│   └── test_services.py    # Service layer tests
└── integration/             # Integration tests
    └── test_api.py         # API endpoint tests
```

### Available Fixtures

- `client`: FastAPI TestClient
- `agent_service`: AgentService instance
- `supervisor_service`: SupervisorService instance
- `workflow_service`: WorkflowService instance
- `sample_agent_data`: Sample agent creation data
- `sample_supervisor_data`: Sample supervisor creation data
- `sample_workflow_data`: Sample workflow creation data

### Code Quality

```bash
# Format code
black app/ tests/

# Check formatting
black --check app/ tests/

# Lint code
ruff check app/ tests/
```

## Frontend Testing

### Setup

```bash
cd web/client
npm install
```

### Unit Tests (Jasmine/Karma)

```bash
# Run unit tests (interactive)
npm test

# Run tests in headless mode (CI)
npm run test:headless

# Generate coverage report
npm run test:headless
# Coverage report: ./coverage/argentic-web-client/index.html
```

### E2E Tests (Playwright)

```bash
# Install Playwright browsers
npx playwright install

# Run E2E tests (headless)
npm run e2e

# Run E2E tests (headed mode - visible browser)
npm run e2e:headed

# Run E2E tests in UI mode (interactive)
npm run e2e:ui

# Run specific test file
npx playwright test e2e/dashboard.spec.ts

# Debug tests
npx playwright test --debug
```

### Test Structure

```
web/client/
├── src/app/
│   ├── components/
│   │   ├── agents/
│   │   │   ├── agents.component.ts
│   │   │   └── agents.component.spec.ts    # Unit tests
│   │   └── dashboard/
│   │       ├── dashboard.component.ts
│   │       └── dashboard.component.spec.ts
│   └── services/
│       ├── api.service.ts
│       └── api.service.spec.ts
└── e2e/                                     # E2E tests
    ├── dashboard.spec.ts
    ├── agents.spec.ts
    └── navigation.spec.ts
```

### Writing Tests

#### Unit Test Example

```typescript
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { MyComponent } from './my.component';

describe('MyComponent', () => {
  let component: MyComponent;
  let fixture: ComponentFixture<MyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MyComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(MyComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
```

#### E2E Test Example

```typescript
import { test, expect } from '@playwright/test';

test('should display page title', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('h1')).toContainText('Dashboard');
});
```

## CI/CD Integration

Tests run automatically on GitHub Actions for:

- Every push to `main` or `develop` branches
- Every pull request
- Manual workflow dispatch

### Workflow Jobs

1. **backend-tests**: Runs Python unit and integration tests
2. **frontend-unit-tests**: Runs Angular unit tests with Karma
3. **frontend-e2e-tests**: Runs Playwright E2E tests
4. **type-generation-check**: Verifies TypeScript models are up to date
5. **build**: Tests production build

### Viewing Results

- Check the **Actions** tab in GitHub
- Coverage reports are uploaded to Codecov
- Playwright reports are available as artifacts

## TypeScript Model Generation

Ensure TypeScript models match Python models:

```bash
# Generate models
cd web/server
python scripts/generate_typescript_models.py

# Or from client directory
cd web/client
npm run generate-models

# Verify models are up to date
git status src/app/models/generated.ts
```

The CI checks if generated models are up to date and fails if they're not.

## Test Coverage

### Target Coverage

- Backend: 80%+
- Frontend: 70%+

### Viewing Coverage

**Backend:**
```bash
cd web/server
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

**Frontend:**
```bash
cd web/client
npm run test:headless
open coverage/argentic-web-client/index.html
```

## Best Practices

### Backend Testing

1. **Use fixtures** for common test data
2. **Test edge cases** and error conditions
3. **Mock external dependencies** (LLM providers, messaging)
4. **Use async tests** for async code
5. **Test API contracts** (request/response schemas)

### Frontend Testing

1. **Test user interactions** (clicks, form inputs)
2. **Mock HTTP calls** using HttpClientTestingModule
3. **Test routing** and navigation
4. **Test component lifecycle** hooks
5. **Use Playwright's auto-waiting** in E2E tests

## Troubleshooting

### Backend Tests

**Issue**: Import errors
```bash
# Solution: Ensure you're in the venv
source .venv/bin/activate
```

**Issue**: Database/connection errors
```bash
# Solution: Check if any services are running that tests depend on
# For now, tests use in-memory storage
```

### Frontend Tests

**Issue**: Chrome not found
```bash
# Solution: Install Playwright browsers
npx playwright install chromium
```

**Issue**: Port already in use
```bash
# Solution: Stop the dev server or use a different port
# Check what's using port 4200
lsof -i :4200
```

**Issue**: Tests timeout
```bash
# Solution: Increase timeout in playwright.config.ts
timeout: 60000  // 60 seconds
```

## Running All Tests

To run the complete test suite locally:

```bash
# Backend tests
cd web/server
source .venv/bin/activate
pytest --cov=app
black --check app/ tests/
ruff check app/ tests/

# Frontend unit tests
cd web/client
npm run test:headless

# Frontend E2E tests (requires backend running)
# Terminal 1: Start backend
cd web/server && source .venv/bin/activate && python main.py

# Terminal 2: Run E2E tests
cd web/client && npm run e2e
```

## Continuous Integration

The GitHub Actions workflow (`.github/workflows/web-tests.yml`) runs:

- Matrix testing for Python 3.11 and 3.12
- All test suites in parallel where possible
- Code quality checks (formatting, linting)
- Build verification
- Coverage reporting

Tests must pass before merging pull requests.

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [Angular Testing Guide](https://angular.dev/guide/testing)
- [Playwright Documentation](https://playwright.dev/)
- [Karma Configuration](https://karma-runner.github.io/latest/config/configuration-file.html)
