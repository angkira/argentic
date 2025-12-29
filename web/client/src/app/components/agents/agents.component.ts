import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, merge, EMPTY } from 'rxjs';
import { switchMap, startWith, catchError, map, tap, shareReplay } from 'rxjs/operators';
import { ApiService } from '../../services/api.service';
import { Agent, AgentCreate } from '../../models';

interface AgentsState {
  agents: Agent[];
  loading: boolean;
  error: string | null;
}

@Component({
  selector: 'app-agents',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './agents.component.html',
  styleUrls: ['./agents.component.scss']
})
export class AgentsComponent {
  // Signals for local UI state only
  showCreateModal = signal(false);

  // Action subjects for user interactions
  private refreshTrigger$ = new Subject<void>();
  private createAction$ = new Subject<AgentCreate>();
  private deleteAction$ = new Subject<string>();
  private toggleAction$ = new Subject<Agent>();

  // Default form values
  readonly defaultAgent: AgentCreate = {
    role: '',
    description: '',
    system_prompt: '',
    expected_output_format: 'json',
    enable_dialogue_logging: false,
    max_consecutive_tool_calls: 3,
    max_dialogue_history_items: 100,
    max_context_iterations: 10,
    enable_adaptive_context_management: true
  };

  newAgent: AgentCreate = { ...this.defaultAgent };

  // Declarative streams
  private readonly createEffect$ = this.createAction$.pipe(
    switchMap(agent =>
      this.apiService.createAgent(agent).pipe(
        tap(() => {
          this.closeCreateModal();
          this.refreshTrigger$.next();
        }),
        catchError(error => {
          console.error('Error creating agent:', error);
          alert('Failed to create agent');
          return EMPTY;
        })
      )
    ),
    takeUntilDestroyed()
  );

  private readonly deleteEffect$ = this.deleteAction$.pipe(
    switchMap(id =>
      this.apiService.deleteAgent(id).pipe(
        tap(() => this.refreshTrigger$.next()),
        catchError(error => {
          console.error('Error deleting agent:', error);
          return EMPTY;
        })
      )
    ),
    takeUntilDestroyed()
  );

  private readonly toggleEffect$ = this.toggleAction$.pipe(
    switchMap(agent => {
      const action$ = agent.status === 'running'
        ? this.apiService.stopAgent(agent.id)
        : this.apiService.startAgent(agent.id);

      return action$.pipe(
        tap(() => this.refreshTrigger$.next()),
        catchError(error => {
          console.error('Error toggling agent:', error);
          return EMPTY;
        })
      );
    }),
    takeUntilDestroyed()
  );

  // Main data stream with automatic refresh on actions
  readonly state$ = merge(
    this.refreshTrigger$.pipe(startWith(void 0)),
    this.createEffect$,
    this.deleteEffect$,
    this.toggleEffect$
  ).pipe(
    switchMap(() =>
      this.apiService.getAgents().pipe(
        map(agents => ({
          agents,
          loading: false,
          error: null
        } as AgentsState)),
        startWith({ agents: [], loading: true, error: null } as AgentsState),
        catchError(error => {
          console.error('Error loading agents:', error);
          return [{ agents: [], loading: false, error: error.message } as AgentsState];
        })
      )
    ),
    shareReplay({ bufferSize: 1, refCount: true }),
    takeUntilDestroyed()
  );

  constructor(private apiService: ApiService) {
    // Trigger initial load
    this.refreshTrigger$.next();
  }

  openCreateModal(): void {
    this.showCreateModal.set(true);
  }

  closeCreateModal(): void {
    this.showCreateModal.set(false);
    this.newAgent = { ...this.defaultAgent };
  }

  createAgent(): void {
    this.createAction$.next(this.newAgent);
  }

  deleteAgent(id: string): void {
    if (confirm('Are you sure you want to delete this agent?')) {
      this.deleteAction$.next(id);
    }
  }

  toggleAgent(agent: Agent): void {
    this.toggleAction$.next(agent);
  }

  getStatusClass(status: string): string {
    switch (status) {
      case 'running': return 'badge-success';
      case 'stopped': return 'badge-secondary';
      case 'error': return 'badge-danger';
      default: return 'badge-secondary';
    }
  }
}
