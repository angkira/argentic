import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { Supervisor, SupervisorCreate } from '../../models';

@Component({
  selector: 'app-supervisors',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './supervisors.component.html',
  styleUrls: ['./supervisors.component.scss']
})
export class SupervisorsComponent implements OnInit {
  supervisors: Supervisor[] = [];
  loading = true;
  showCreateModal = false;

  newSupervisor: SupervisorCreate = {
    role: '',
    description: '',
    system_prompt: '',
    worker_agents: [],
    enable_dialogue_logging: true,
    max_dialogue_history_items: 100
  };

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    this.loadSupervisors();
  }

  loadSupervisors(): void {
    this.loading = true;
    this.apiService.getSupervisors().subscribe({
      next: (supervisors) => {
        this.supervisors = supervisors;
        this.loading = false;
      },
      error: (error) => {
        console.error('Error loading supervisors:', error);
        this.loading = false;
      }
    });
  }

  openCreateModal(): void {
    this.showCreateModal = true;
  }

  closeCreateModal(): void {
    this.showCreateModal = false;
  }

  createSupervisor(): void {
    this.apiService.createSupervisor(this.newSupervisor).subscribe({
      next: () => {
        this.closeCreateModal();
        this.loadSupervisors();
      },
      error: (error) => {
        console.error('Error creating supervisor:', error);
        alert('Failed to create supervisor');
      }
    });
  }

  deleteSupervisor(id: string): void {
    if (confirm('Are you sure you want to delete this supervisor?')) {
      this.apiService.deleteSupervisor(id).subscribe({
        next: () => this.loadSupervisors(),
        error: (error) => console.error('Error deleting supervisor:', error)
      });
    }
  }

  toggleSupervisor(supervisor: Supervisor): void {
    if (supervisor.status === 'running') {
      this.apiService.stopSupervisor(supervisor.id).subscribe({
        next: () => this.loadSupervisors(),
        error: (error) => console.error('Error stopping supervisor:', error)
      });
    } else {
      this.apiService.startSupervisor(supervisor.id).subscribe({
        next: () => this.loadSupervisors(),
        error: (error) => console.error('Error starting supervisor:', error)
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
