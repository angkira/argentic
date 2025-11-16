import { Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { AgentsComponent } from './components/agents/agents.component';
import { SupervisorsComponent } from './components/supervisors/supervisors.component';
import { WorkflowsComponent } from './components/workflows/workflows.component';
import { WorkflowBuilderComponent } from './components/workflow-builder/workflow-builder.component';
import { ConfigComponent } from './components/config/config.component';

export const routes: Routes = [
  { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'agents', component: AgentsComponent },
  { path: 'supervisors', component: SupervisorsComponent },
  { path: 'workflows', component: WorkflowsComponent },
  { path: 'workflows/:id/edit', component: WorkflowBuilderComponent },
  { path: 'workflows/new', component: WorkflowBuilderComponent },
  { path: 'config', component: ConfigComponent },
  { path: '**', redirectTo: '/dashboard' }
];
