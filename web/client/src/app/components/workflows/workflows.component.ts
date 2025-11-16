import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { ApiService } from '../../services/api.service';
import { Workflow } from '../../models';

@Component({
  selector: 'app-workflows',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './workflows.component.html',
  styleUrls: ['./workflows.component.scss']
})
export class WorkflowsComponent implements OnInit {
  workflows: Workflow[] = [];
  loading = true;

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    this.loadWorkflows();
  }

  loadWorkflows(): void {
    this.loading = true;
    this.apiService.getWorkflows().subscribe({
      next: (workflows) => {
        this.workflows = workflows;
        this.loading = false;
      },
      error: (error) => {
        console.error('Error loading workflows:', error);
        this.loading = false;
      }
    });
  }

  deleteWorkflow(id: string): void {
    if (confirm('Are you sure you want to delete this workflow?')) {
      this.apiService.deleteWorkflow(id).subscribe({
        next: () => this.loadWorkflows(),
        error: (error) => console.error('Error deleting workflow:', error)
      });
    }
  }

  toggleWorkflow(workflow: Workflow): void {
    if (workflow.status === 'running') {
      this.apiService.stopWorkflow(workflow.id).subscribe({
        next: () => this.loadWorkflows(),
        error: (error) => console.error('Error stopping workflow:', error)
      });
    } else {
      this.apiService.startWorkflow(workflow.id).subscribe({
        next: () => this.loadWorkflows(),
        error: (error) => console.error('Error starting workflow:', error)
      });
    }
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
