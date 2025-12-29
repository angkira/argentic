# Argentic Web Client

Angular 20 frontend for the Argentic visual agent builder.

## Installation

```bash
npm install
```

## Development Server

```bash
npm start
```

Navigate to `http://localhost:4200/`. The application will automatically reload if you change any of the source files.

## Build

```bash
npm run build
```

Build artifacts will be stored in the `dist/` directory.

## Testing

```bash
npm test
```

## Linting

```bash
npm run lint
```

## Project Structure

```
client/
├── src/
│   ├── app/
│   │   ├── components/        # Feature components
│   │   │   ├── dashboard/     # Dashboard view
│   │   │   ├── agents/        # Agent management
│   │   │   ├── supervisors/   # Supervisor management
│   │   │   ├── workflows/     # Workflow list
│   │   │   ├── workflow-builder/  # Visual workflow editor
│   │   │   └── config/        # Configuration settings
│   │   ├── models/            # TypeScript interfaces
│   │   ├── services/          # API services
│   │   ├── core/              # Core utilities
│   │   ├── app.component.ts   # Root component
│   │   ├── app.config.ts      # App configuration
│   │   └── app.routes.ts      # Route definitions
│   ├── assets/                # Static assets
│   ├── styles.scss            # Global styles
│   └── index.html             # Main HTML
├── angular.json               # Angular CLI configuration
├── tsconfig.json              # TypeScript configuration
├── proxy.conf.json            # Dev server proxy config
└── package.json               # Dependencies
```

## Angular 20 Features

This project uses the latest Angular 20 features:

### Standalone Components

All components are standalone (no NgModules):

```typescript
@Component({
  selector: 'app-example',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './example.component.html'
})
export class ExampleComponent {}
```

### New Control Flow Syntax

Using Angular's new built-in control flow:

```html
<!-- @if instead of *ngIf -->
@if (condition) {
  <div>Content</div>
}

<!-- @for instead of *ngFor -->
@for (item of items; track item.id) {
  <div>{{ item.name }}</div>
}

<!-- @switch instead of *ngSwitch -->
@switch (value) {
  @case ('option1') { <div>Option 1</div> }
  @case ('option2') { <div>Option 2</div> }
  @default { <div>Default</div> }
}
```

### Signals (Future Enhancement)

The project is ready to adopt Angular Signals for reactive state management:

```typescript
import { signal, computed } from '@angular/core';

export class Component {
  count = signal(0);
  doubleCount = computed(() => this.count() * 2);

  increment() {
    this.count.update(n => n + 1);
  }
}
```

## TypeScript Model Generation

Generate TypeScript models from the backend:

```bash
npm run generate-models
```

This runs the Python script from the server directory and updates `src/app/models/generated.ts`.

## API Integration

The client uses a proxy configuration (`proxy.conf.json`) to forward API requests to the backend during development:

```json
{
  "/api": {
    "target": "http://localhost:8000",
    "secure": false,
    "changeOrigin": true
  }
}
```

All `/api/*` requests are forwarded to the FastAPI backend.

## Components

### Dashboard

Overview of all agents, supervisors, and workflows with quick actions.

**Route**: `/dashboard`

### Agents

Create, view, update, and delete agents. Configure agent parameters like system prompts, output formats, and tool call limits.

**Route**: `/agents`

### Supervisors

Manage supervisor agents that coordinate multiple worker agents.

**Route**: `/supervisors`

### Workflows

List and manage workflows. Each workflow can be visually edited in the workflow builder.

**Route**: `/workflows`

### Workflow Builder

Visual drag-and-drop interface for creating agent workflows (similar to n8n).

**Routes**:
- `/workflows/new` - Create new workflow
- `/workflows/:id/edit` - Edit existing workflow

### Configuration

Configure LLM providers and messaging backends.

**Route**: `/config`

## Services

### ApiService

Handles all HTTP communication with the backend:

```typescript
import { ApiService } from './services/api.service';

constructor(private api: ApiService) {}

ngOnInit() {
  this.api.getAgents().subscribe(agents => {
    this.agents = agents;
  });
}
```

## Styling

The project uses SCSS with a utility-first approach. Global styles are in `src/styles.scss`:

- CSS variables for theming
- Reusable component styles
- Utility classes for spacing, layout, etc.

## Building for Production

```bash
npm run build
```

The build artifacts will be stored in `dist/argentic-web-client/`. Serve with any static file server:

```bash
# Using a simple HTTP server
npx http-server dist/argentic-web-client -p 8080
```

For production deployment, consider:

- Nginx or Apache for serving static files
- Environment-specific configurations
- API proxy configuration
- SSL/TLS certificates

## Environment Configuration

Create `src/environments/environment.prod.ts` for production settings:

```typescript
export const environment = {
  production: true,
  apiUrl: 'https://api.example.com'
};
```

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

MIT
