import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, merge, EMPTY } from 'rxjs';
import { switchMap, startWith, catchError, map, tap, shareReplay } from 'rxjs/operators';
import { ApiService } from '../../services/api.service';
import { Workflow } from '../../models';

interface WorkflowsState {
  workflows: Workflow[];
  loading: boolean;
  error: string | null;
}

@Component({
  selector: 'app-workflows',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './workflows.component.html',
  styleUrls: ['./workflows.component.scss']
})
export class WorkflowsComponent {
  // Action subjects for user interactions
  private refreshTrigger$ = new Subject<void>();
  private deleteAction$ = new Subject<string>();
  private toggleAction$ = new Subject<Workflow>();

  // Declarative streams
  private readonly deleteEffect$ = this.deleteAction$.pipe(
    switchMap(id =>
      this.apiService.deleteWorkflow(id).pipe(
        tap(() => this.refreshTrigger$.next()),
        catchError(error => {
          console.error('Error deleting workflow:', error);
          return EMPTY;
        })
      )
    ),
    takeUntilDestroyed()
  );

  private readonly toggleEffect$ = this.toggleAction$.pipe(
    switchMap(workflow => {
      const action$ = workflow.status === 'running'
        ? this.apiService.stopWorkflow(workflow.id)
        : this.apiService.startWorkflow(workflow.id);

      return action$.pipe(
        tap(() => this.refreshTrigger$.next()),
        catchError(error => {
          console.error('Error toggling workflow:', error);
          return EMPTY;
        })
      );
    }),
    takeUntilDestroyed()
  );

  // Main data stream with automatic refresh on actions
  readonly state$ = merge(
    this.refreshTrigger$.pipe(startWith(void 0)),
    this.deleteEffect$,
    this.toggleEffect$
  ).pipe(
    switchMap(() =>
      this.apiService.getWorkflows().pipe(
        map(workflows => ({
          workflows,
          loading: false,
          error: null
        } as WorkflowsState)),
        startWith({ workflows: [], loading: true, error: null } as WorkflowsState),
        catchError(error => {
          console.error('Error loading workflows:', error);
          return [{ workflows: [], loading: false, error: error.message } as WorkflowsState];
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

  deleteWorkflow(id: string): void {
    if (confirm('Are you sure you want to delete this workflow?')) {
      this.deleteAction$.next(id);
    }
  }

  toggleWorkflow(workflow: Workflow): void {
    this.toggleAction$.next(workflow);
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
