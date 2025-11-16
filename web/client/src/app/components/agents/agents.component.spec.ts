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
    fixture = TestBed.createComponent(AgentsComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should load agents on init', () => {
    const mockAgents = [
      {
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
      }
    ];

    apiService.getAgents.and.returnValue(of(mockAgents));

    component.ngOnInit();

    expect(apiService.getAgents).toHaveBeenCalled();
    expect(component.agents).toEqual(mockAgents);
    expect(component.loading).toBe(false);
  });

  it('should handle error when loading agents', () => {
    apiService.getAgents.and.returnValue(throwError(() => new Error('API Error')));
    spyOn(console, 'error');

    component.ngOnInit();

    expect(console.error).toHaveBeenCalled();
    expect(component.loading).toBe(false);
  });

  it('should open and close create modal', () => {
    expect(component.showCreateModal).toBe(false);

    component.openCreateModal();
    expect(component.showCreateModal).toBe(true);

    component.closeCreateModal();
    expect(component.showCreateModal).toBe(false);
  });

  it('should create agent', () => {
    const newAgent = {
      id: '1',
      role: 'new_agent',
      description: 'New agent',
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

    apiService.createAgent.and.returnValue(of(newAgent));
    apiService.getAgents.and.returnValue(of([newAgent]));

    component.newAgent.role = 'new_agent';
    component.newAgent.description = 'New agent';
    component.createAgent();

    expect(apiService.createAgent).toHaveBeenCalled();
    expect(component.showCreateModal).toBe(false);
  });

  it('should get correct status class', () => {
    expect(component.getStatusClass('running')).toBe('badge-success');
    expect(component.getStatusClass('stopped')).toBe('badge-secondary');
    expect(component.getStatusClass('error')).toBe('badge-danger');
  });

  it('should toggle agent status', () => {
    const runningAgent = {
      id: '1',
      role: 'test_agent',
      description: 'Test',
      expected_output_format: 'json' as const,
      enable_dialogue_logging: false,
      max_consecutive_tool_calls: 3,
      max_dialogue_history_items: 100,
      max_context_iterations: 10,
      enable_adaptive_context_management: true,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
      status: 'running' as const
    };

    apiService.stopAgent.and.returnValue(of({ ...runningAgent, status: 'stopped' }));
    apiService.getAgents.and.returnValue(of([]));

    component.toggleAgent(runningAgent);

    expect(apiService.stopAgent).toHaveBeenCalledWith('1');
  });
});
