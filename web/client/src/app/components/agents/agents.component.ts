import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { Agent, AgentCreate } from '../../models';

@Component({
  selector: 'app-agents',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './agents.component.html',
  styleUrls: ['./agents.component.scss']
})
export class AgentsComponent implements OnInit {
  agents: Agent[] = [];
  loading = true;
  showCreateModal = false;

  newAgent: AgentCreate = {
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

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    this.loadAgents();
  }

  loadAgents(): void {
    this.loading = true;
    this.apiService.getAgents().subscribe({
      next: (agents) => {
        this.agents = agents;
        this.loading = false;
      },
      error: (error) => {
        console.error('Error loading agents:', error);
        this.loading = false;
      }
    });
  }

  openCreateModal(): void {
    this.showCreateModal = true;
  }

  closeCreateModal(): void {
    this.showCreateModal = false;
    this.resetNewAgent();
  }

  createAgent(): void {
    this.apiService.createAgent(this.newAgent).subscribe({
      next: () => {
        this.closeCreateModal();
        this.loadAgents();
      },
      error: (error) => {
        console.error('Error creating agent:', error);
        alert('Failed to create agent');
      }
    });
  }

  deleteAgent(id: string): void {
    if (confirm('Are you sure you want to delete this agent?')) {
      this.apiService.deleteAgent(id).subscribe({
        next: () => this.loadAgents(),
        error: (error) => console.error('Error deleting agent:', error)
      });
    }
  }

  toggleAgent(agent: Agent): void {
    if (agent.status === 'running') {
      this.apiService.stopAgent(agent.id).subscribe({
        next: () => this.loadAgents(),
        error: (error) => console.error('Error stopping agent:', error)
      });
    } else {
      this.apiService.startAgent(agent.id).subscribe({
        next: () => this.loadAgents(),
        error: (error) => console.error('Error starting agent:', error)
      });
    }
  }

  private resetNewAgent(): void {
    this.newAgent = {
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
