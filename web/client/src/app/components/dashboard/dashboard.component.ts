import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { combineLatest } from 'rxjs';
import { map, startWith, catchError, shareReplay } from 'rxjs/operators';
import { ApiService } from '../../services/api.service';
import { Agent, Supervisor, Workflow } from '../../models';

interface DashboardState {
  agents: Agent[];
  supervisors: Supervisor[];
  workflows: Workflow[];
  runningAgents: number;
  runningSupervisors: number;
  runningWorkflows: number;
  loading: boolean;
  error: string | null;
}

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent {
  // Declarative data stream combining all dashboard data
  readonly state$ = combineLatest({
    agents: this.apiService.getAgents(),
    supervisors: this.apiService.getSupervisors(),
    workflows: this.apiService.getWorkflows()
  }).pipe(
    map(({ agents, supervisors, workflows }) => ({
      agents,
      supervisors,
      workflows,
      runningAgents: agents.filter(a => a.status === 'running').length,
      runningSupervisors: supervisors.filter(s => s.status === 'running').length,
      runningWorkflows: workflows.filter(w => w.status === 'running').length,
      loading: false,
      error: null
    } as DashboardState)),
    startWith({
      agents: [],
      supervisors: [],
      workflows: [],
      runningAgents: 0,
      runningSupervisors: 0,
      runningWorkflows: 0,
      loading: true,
      error: null
    } as DashboardState),
    catchError(error => {
      console.error('Error loading dashboard data:', error);
      return [{
        agents: [],
        supervisors: [],
        workflows: [],
        runningAgents: 0,
        runningSupervisors: 0,
        runningWorkflows: 0,
        loading: false,
        error: error.message
      } as DashboardState];
    }),
    shareReplay({ bufferSize: 1, refCount: true }),
    takeUntilDestroyed()
  );

  constructor(private apiService: ApiService) {}
}
