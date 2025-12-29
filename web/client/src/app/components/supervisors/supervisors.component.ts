import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, merge, EMPTY } from 'rxjs';
import { switchMap, startWith, catchError, map, tap, shareReplay } from 'rxjs/operators';
import { ApiService } from '../../services/api.service';
import { Supervisor, SupervisorCreate } from '../../models';

interface SupervisorsState {
  supervisors: Supervisor[];
  loading: boolean;
  error: string | null;
}

@Component({
  selector: 'app-supervisors',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './supervisors.component.html',
  styleUrls: ['./supervisors.component.scss']
})
export class SupervisorsComponent {
  // Signals for local UI state only
  showCreateModal = signal(false);

  // Action subjects for user interactions
  private refreshTrigger$ = new Subject<void>();
  private createAction$ = new Subject<SupervisorCreate>();
  private deleteAction$ = new Subject<string>();
  private toggleAction$ = new Subject<Supervisor>();

  // Default form values
  readonly defaultSupervisor: SupervisorCreate = {
    role: '',
    description: '',
    system_prompt: '',
    worker_agents: [],
    enable_dialogue_logging: true,
    max_dialogue_history_items: 100
  };

  newSupervisor: SupervisorCreate = { ...this.defaultSupervisor };

  // Declarative streams
  private readonly createEffect$ = this.createAction$.pipe(
    switchMap(supervisor =>
      this.apiService.createSupervisor(supervisor).pipe(
        tap(() => {
          this.closeCreateModal();
          this.refreshTrigger$.next();
        }),
        catchError(error => {
          console.error('Error creating supervisor:', error);
          alert('Failed to create supervisor');
          return EMPTY;
        })
      )
    ),
    takeUntilDestroyed()
  );

  private readonly deleteEffect$ = this.deleteAction$.pipe(
    switchMap(id =>
      this.apiService.deleteSupervisor(id).pipe(
        tap(() => this.refreshTrigger$.next()),
        catchError(error => {
          console.error('Error deleting supervisor:', error);
          return EMPTY;
        })
      )
    ),
    takeUntilDestroyed()
  );

  private readonly toggleEffect$ = this.toggleAction$.pipe(
    switchMap(supervisor => {
      const action$ = supervisor.status === 'running'
        ? this.apiService.stopSupervisor(supervisor.id)
        : this.apiService.startSupervisor(supervisor.id);

      return action$.pipe(
        tap(() => this.refreshTrigger$.next()),
        catchError(error => {
          console.error('Error toggling supervisor:', error);
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
      this.apiService.getSupervisors().pipe(
        map(supervisors => ({
          supervisors,
          loading: false,
          error: null
        } as SupervisorsState)),
        startWith({ supervisors: [], loading: true, error: null } as SupervisorsState),
        catchError(error => {
          console.error('Error loading supervisors:', error);
          return [{ supervisors: [], loading: false, error: error.message } as SupervisorsState];
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
    this.newSupervisor = { ...this.defaultSupervisor };
  }

  createSupervisor(): void {
    this.createAction$.next(this.newSupervisor);
  }

  deleteSupervisor(id: string): void {
    if (confirm('Are you sure you want to delete this supervisor?')) {
      this.deleteAction$.next(id);
    }
  }

  toggleSupervisor(supervisor: Supervisor): void {
    this.toggleAction$.next(supervisor);
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
