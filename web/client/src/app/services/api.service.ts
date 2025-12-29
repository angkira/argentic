import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  Agent,
  AgentCreate,
  AgentUpdate,
  Supervisor,
  SupervisorCreate,
  SupervisorUpdate,
  Workflow,
  WorkflowCreate,
  WorkflowUpdate,
  LLMProviderInfo,
  MessagingProtocolInfo,
  LLMProviderConfig,
  MessagingConfig
} from '../models';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl = '/api';

  constructor(private http: HttpClient) {}

  // Agent endpoints
  getAgents(): Observable<Agent[]> {
    return this.http.get<Agent[]>(`${this.baseUrl}/agents`);
  }

  getAgent(id: string): Observable<Agent> {
    return this.http.get<Agent>(`${this.baseUrl}/agents/${id}`);
  }

  createAgent(agent: AgentCreate): Observable<Agent> {
    return this.http.post<Agent>(`${this.baseUrl}/agents`, agent);
  }

  updateAgent(id: string, agent: AgentUpdate): Observable<Agent> {
    return this.http.patch<Agent>(`${this.baseUrl}/agents/${id}`, agent);
  }

  deleteAgent(id: string): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/agents/${id}`);
  }

  startAgent(id: string): Observable<Agent> {
    return this.http.post<Agent>(`${this.baseUrl}/agents/${id}/start`, {});
  }

  stopAgent(id: string): Observable<Agent> {
    return this.http.post<Agent>(`${this.baseUrl}/agents/${id}/stop`, {});
  }

  // Supervisor endpoints
  getSupervisors(): Observable<Supervisor[]> {
    return this.http.get<Supervisor[]>(`${this.baseUrl}/supervisors`);
  }

  getSupervisor(id: string): Observable<Supervisor> {
    return this.http.get<Supervisor>(`${this.baseUrl}/supervisors/${id}`);
  }

  createSupervisor(supervisor: SupervisorCreate): Observable<Supervisor> {
    return this.http.post<Supervisor>(`${this.baseUrl}/supervisors`, supervisor);
  }

  updateSupervisor(id: string, supervisor: SupervisorUpdate): Observable<Supervisor> {
    return this.http.patch<Supervisor>(`${this.baseUrl}/supervisors/${id}`, supervisor);
  }

  deleteSupervisor(id: string): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/supervisors/${id}`);
  }

  startSupervisor(id: string): Observable<Supervisor> {
    return this.http.post<Supervisor>(`${this.baseUrl}/supervisors/${id}/start`, {});
  }

  stopSupervisor(id: string): Observable<Supervisor> {
    return this.http.post<Supervisor>(`${this.baseUrl}/supervisors/${id}/stop`, {});
  }

  // Workflow endpoints
  getWorkflows(): Observable<Workflow[]> {
    return this.http.get<Workflow[]>(`${this.baseUrl}/workflows`);
  }

  getWorkflow(id: string): Observable<Workflow> {
    return this.http.get<Workflow>(`${this.baseUrl}/workflows/${id}`);
  }

  createWorkflow(workflow: WorkflowCreate): Observable<Workflow> {
    return this.http.post<Workflow>(`${this.baseUrl}/workflows`, workflow);
  }

  updateWorkflow(id: string, workflow: WorkflowUpdate): Observable<Workflow> {
    return this.http.patch<Workflow>(`${this.baseUrl}/workflows/${id}`, workflow);
  }

  deleteWorkflow(id: string): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/workflows/${id}`);
  }

  startWorkflow(id: string): Observable<Workflow> {
    return this.http.post<Workflow>(`${this.baseUrl}/workflows/${id}/start`, {});
  }

  stopWorkflow(id: string): Observable<Workflow> {
    return this.http.post<Workflow>(`${this.baseUrl}/workflows/${id}/stop`, {});
  }

  // Config endpoints
  getLLMProviders(): Observable<LLMProviderInfo[]> {
    return this.http.get<LLMProviderInfo[]>(`${this.baseUrl}/config/llm-providers`);
  }

  getMessagingProtocols(): Observable<MessagingProtocolInfo[]> {
    return this.http.get<MessagingProtocolInfo[]>(`${this.baseUrl}/config/messaging-protocols`);
  }

  validateLLMConfig(config: LLMProviderConfig): Observable<{ valid: boolean; provider: string }> {
    return this.http.post<{ valid: boolean; provider: string }>(
      `${this.baseUrl}/config/llm-providers/validate`,
      config
    );
  }

  validateMessagingConfig(config: MessagingConfig): Observable<{ valid: boolean; protocol: string }> {
    return this.http.post<{ valid: boolean; protocol: string }>(
      `${this.baseUrl}/config/messaging/validate`,
      config
    );
  }
}
