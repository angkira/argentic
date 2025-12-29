import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { ApiService } from '../../services/api.service';
import { Workflow, WorkflowCreate } from '../../models';

@Component({
  selector: 'app-workflow-builder',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './workflow-builder.component.html',
  styleUrls: ['./workflow-builder.component.scss']
})
export class WorkflowBuilderComponent implements OnInit {
  workflow: WorkflowCreate = {
    name: '',
    description: '',
    nodes: [],
    edges: []
  };
  editMode = false;
  workflowId?: string;

  constructor(
    private apiService: ApiService,
    private route: ActivatedRoute,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.workflowId = this.route.snapshot.params['id'];
    if (this.workflowId) {
      this.editMode = true;
      this.loadWorkflow(this.workflowId);
    }
  }

  loadWorkflow(id: string): void {
    this.apiService.getWorkflow(id).subscribe({
      next: (workflow: Workflow) => {
        this.workflow = {
          name: workflow.name,
          description: workflow.description,
          nodes: workflow.nodes,
          edges: workflow.edges
        };
      },
      error: (error) => console.error('Error loading workflow:', error)
    });
  }

  save(): void {
    if (this.editMode && this.workflowId) {
      this.apiService.updateWorkflow(this.workflowId, this.workflow).subscribe({
        next: () => this.router.navigate(['/workflows']),
        error: (error) => console.error('Error updating workflow:', error)
      });
    } else {
      this.apiService.createWorkflow(this.workflow).subscribe({
        next: () => this.router.navigate(['/workflows']),
        error: (error) => console.error('Error creating workflow:', error)
      });
    }
  }

  cancel(): void {
    this.router.navigate(['/workflows']);
  }
}
