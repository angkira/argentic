import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { FormsModule } from '@angular/forms';
import { AgentsComponent } from './agents.component';
import { ApiService } from '../../services/api.service';
import { of, throwError } from 'rxjs';

describe('AgentsComponent', () => {
  let component: AgentsComponent;
  let fixture: ComponentFixture<AgentsComponent>;
  let apiService: jasmine.SpyObj<ApiService>;

  const mockAgent = {
    id: '1',
    role: 'test_agent',
    description: 'Test agent',
    expected_output_format: 'json' as const,
    enable_dialogue_logging: false,
    max_consecutive_tool_calls: 3,
    max_dialogue_history_items: 100,
    max_context_iterations: 10,
    enable_adaptive_context_management: true,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    status: 'stopped' as const
  };

  beforeEach(async () => {
    const apiServiceSpy = jasmine.createSpyObj('ApiService', [
      'getAgents',
      'createAgent',
      'deleteAgent',
      'startAgent',
      'stopAgent'
    ]);

    await TestBed.configureTestingModule({
      imports: [AgentsComponent, HttpClientTestingModule, FormsModule],
      providers: [{ provide: ApiService, useValue: apiServiceSpy }]
    }).compileComponents();

    apiService = TestBed.inject(ApiService) as jasmine.SpyObj<ApiService>;

    // Set default return values for all API methods
    apiService.getAgents.and.returnValue(of([]));
    apiService.createAgent.and.returnValue(of(mockAgent));
    apiService.deleteAgent.and.returnValue(of(void 0));
    apiService.startAgent.and.returnValue(of(mockAgent));
    apiService.stopAgent.and.returnValue(of(mockAgent));

    fixture = TestBed.createComponent(AgentsComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should have state$ observable that emits initial loading state', (done) => {
    component.state$.subscribe(state => {
      if (state.loading) {
        expect(state.agents).toEqual([]);
        expect(state.loading).toBe(true);
        expect(state.error).toBeNull();
        done();
      }
    });
  });

  it('should load agents via state$ stream', (done) => {
    const mockAgents = [mockAgent];
    apiService.getAgents.and.returnValue(of(mockAgents));

    // Create new component instance to trigger stream
    fixture = TestBed.createComponent(AgentsComponent);
    component = fixture.componentInstance;

    // Skip loading state and get data state
    let skipFirst = true;
    component.state$.subscribe(state => {
      if (skipFirst) {
        skipFirst = false;
        return;
      }
      if (!state.loading) {
        expect(state.agents).toEqual(mockAgents);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
        expect(apiService.getAgents).toHaveBeenCalled();
        done();
      }
    });
  });

  it('should handle error when loading agents', (done) => {
    apiService.getAgents.and.returnValue(throwError(() => new Error('API Error')));
    spyOn(console, 'error');

    // Create new component instance to trigger stream
    fixture = TestBed.createComponent(AgentsComponent);
    component = fixture.componentInstance;

    let skipFirst = true;
    component.state$.subscribe(state => {
      if (skipFirst) {
        skipFirst = false;
        return;
      }
      if (!state.loading && state.error) {
        expect(state.error).toBe('API Error');
        expect(state.agents).toEqual([]);
        expect(console.error).toHaveBeenCalled();
        done();
      }
    });
  });

  it('should toggle create modal using signal', () => {
    expect(component.showCreateModal()).toBe(false);

    component.openCreateModal();
    expect(component.showCreateModal()).toBe(true);

    component.closeCreateModal();
    expect(component.showCreateModal()).toBe(false);
  });

  it('should reset form when closing modal', () => {
    component.newAgent.role = 'test';
    component.newAgent.description = 'test desc';

    component.closeCreateModal();

    expect(component.newAgent.role).toBe('');
    expect(component.newAgent.description).toBe('');
  });

  it('should create agent and refresh list', (done) => {
    const createdAgent = { ...mockAgent, id: '2', role: 'new_agent' };
    apiService.createAgent.and.returnValue(of(createdAgent));
    apiService.getAgents.and.returnValue(of([createdAgent]));

    component.newAgent.role = 'new_agent';
    component.newAgent.description = 'New agent';
    component.showCreateModal.set(true);

    component.createAgent();

    // Modal should close
    expect(component.showCreateModal()).toBe(false);
    expect(apiService.createAgent).toHaveBeenCalled();

    // Wait for state to update
    setTimeout(() => {
      expect(apiService.getAgents).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should delete agent and refresh list', (done) => {
    spyOn(window, 'confirm').and.returnValue(true);
    apiService.deleteAgent.and.returnValue(of(void 0));

    component.deleteAgent('1');

    expect(apiService.deleteAgent).toHaveBeenCalledWith('1');

    setTimeout(() => {
      expect(apiService.getAgents).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should not delete agent if user cancels', () => {
    spyOn(window, 'confirm').and.returnValue(false);

    component.deleteAgent('1');

    expect(apiService.deleteAgent).not.toHaveBeenCalled();
  });

  it('should stop running agent', (done) => {
    const runningAgent = { ...mockAgent, status: 'running' as const };
    apiService.stopAgent.and.returnValue(of({ ...runningAgent, status: 'stopped' as const }));

    component.toggleAgent(runningAgent);

    expect(apiService.stopAgent).toHaveBeenCalledWith('1');

    setTimeout(() => {
      expect(apiService.getAgents).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should start stopped agent', (done) => {
    const stoppedAgent = { ...mockAgent, status: 'stopped' as const };
    apiService.startAgent.and.returnValue(of({ ...stoppedAgent, status: 'running' as const }));

    component.toggleAgent(stoppedAgent);

    expect(apiService.startAgent).toHaveBeenCalledWith('1');

    setTimeout(() => {
      expect(apiService.getAgents).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should get correct status class', () => {
    expect(component.getStatusClass('running')).toBe('badge-success');
    expect(component.getStatusClass('stopped')).toBe('badge-secondary');
    expect(component.getStatusClass('error')).toBe('badge-danger');
    expect(component.getStatusClass('unknown')).toBe('badge-secondary');
  });

  it('should handle create error gracefully', () => {
    spyOn(window, 'alert');
    spyOn(console, 'error');
    apiService.createAgent.and.returnValue(throwError(() => new Error('Create failed')));

    component.newAgent.role = 'test';
    component.newAgent.description = 'test';
    component.createAgent();

    setTimeout(() => {
      expect(console.error).toHaveBeenCalled();
      expect(window.alert).toHaveBeenCalledWith('Failed to create agent');
    }, 100);
  });

  it('should have default agent values', () => {
    expect(component.defaultAgent.expected_output_format).toBe('json');
    expect(component.defaultAgent.enable_dialogue_logging).toBe(false);
    expect(component.defaultAgent.max_consecutive_tool_calls).toBe(3);
    expect(component.defaultAgent.max_dialogue_history_items).toBe(100);
    expect(component.defaultAgent.max_context_iterations).toBe(10);
    expect(component.defaultAgent.enable_adaptive_context_management).toBe(true);
  });
});
