import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { ApiService } from '../../services/api.service';
import { Agent, Supervisor, Workflow } from '../../models';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit {
  agents: Agent[] = [];
  supervisors: Supervisor[] = [];
  workflows: Workflow[] = [];
  loading = true;

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    this.loadData();
  }

  loadData(): void {
    this.loading = true;
    Promise.all([
      this.apiService.getAgents().toPromise(),
      this.apiService.getSupervisors().toPromise(),
      this.apiService.getWorkflows().toPromise()
    ]).then(([agents, supervisors, workflows]) => {
      this.agents = agents || [];
      this.supervisors = supervisors || [];
      this.workflows = workflows || [];
      this.loading = false;
    }).catch(error => {
      console.error('Error loading dashboard data:', error);
      this.loading = false;
    });
  }

  get runningAgents(): number {
    return this.agents.filter(a => a.status === 'running').length;
  }

  get runningSupervisors(): number {
    return this.supervisors.filter(s => s.status === 'running').length;
  }

  get runningWorkflows(): number {
    return this.workflows.filter(w => w.status === 'running').length;
  }
}
