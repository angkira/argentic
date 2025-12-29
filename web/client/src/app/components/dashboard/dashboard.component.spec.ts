import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { RouterTestingModule } from '@angular/router/testing';
import { DashboardComponent } from './dashboard.component';
import { ApiService } from '../../services/api.service';
import { of, throwError } from 'rxjs';
import { Agent, Supervisor, Workflow } from '../../models';

describe('DashboardComponent', () => {
  let component: DashboardComponent;
  let fixture: ComponentFixture<DashboardComponent>;
  let apiService: jasmine.SpyObj<ApiService>;

  const mockAgents: Agent[] = [
    {
      id: 'agent-1',
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
    },
    {
      id: 'agent-2',
      role: 'stopped_agent',
      description: 'Test',
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

  const mockSupervisors: Supervisor[] = [
    {
      id: 'supervisor-1',
      role: 'test_supervisor',
      description: 'Test',
      system_prompt: '',
      worker_agents: [],
      enable_dialogue_logging: true,
      max_dialogue_history_items: 100,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
      status: 'running' as const
    },
    {
      id: 'supervisor-2',
      role: 'stopped_supervisor',
      description: 'Test',
      system_prompt: '',
      worker_agents: [],
      enable_dialogue_logging: true,
      max_dialogue_history_items: 100,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
      status: 'stopped' as const
    }
  ];

  const mockWorkflows: Workflow[] = [
    {
      id: 'workflow-1',
      name: 'test_workflow',
      description: 'Test',
      nodes: [],
      edges: [],
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
      status: 'stopped' as const
    }
  ];

  beforeEach(async () => {
    const apiServiceSpy = jasmine.createSpyObj('ApiService', [
      'getAgents',
      'getSupervisors',
      'getWorkflows'
    ]);

    await TestBed.configureTestingModule({
      imports: [DashboardComponent, HttpClientTestingModule, RouterTestingModule],
      providers: [{ provide: ApiService, useValue: apiServiceSpy }]
    }).compileComponents();

    apiService = TestBed.inject(ApiService) as jasmine.SpyObj<ApiService>;
    apiService.getAgents.and.returnValue(of(mockAgents));
    apiService.getSupervisors.and.returnValue(of(mockSupervisors));
    apiService.getWorkflows.and.returnValue(of(mockWorkflows));

    fixture = TestBed.createComponent(DashboardComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should have state$ observable that emits initial loading state', (done) => {
    component.state$.subscribe(state => {
      if (state.loading) {
        expect(state.agents).toEqual([]);
        expect(state.supervisors).toEqual([]);
        expect(state.workflows).toEqual([]);
        expect(state.runningAgents).toBe(0);
        expect(state.runningSupervisors).toBe(0);
        expect(state.runningWorkflows).toBe(0);
        expect(state.loading).toBe(true);
        expect(state.error).toBeNull();
        done();
      }
    });
  });

  it('should emit dashboard data after loading', (done) => {
    let emissionCount = 0;
    component.state$.subscribe(state => {
      emissionCount++;
      if (emissionCount === 2 && !state.loading) {
        expect(state.agents).toEqual(mockAgents);
        expect(state.supervisors).toEqual(mockSupervisors);
        expect(state.workflows).toEqual(mockWorkflows);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
        done();
      }
    });
  });

  it('should calculate running agents correctly', (done) => {
    let emissionCount = 0;
    component.state$.subscribe(state => {
      emissionCount++;
      if (emissionCount === 2 && !state.loading) {
        // mockAgents has 1 running agent
        expect(state.runningAgents).toBe(1);
        done();
      }
    });
  });

  it('should calculate running supervisors correctly', (done) => {
    let emissionCount = 0;
    component.state$.subscribe(state => {
      emissionCount++;
      if (emissionCount === 2 && !state.loading) {
        // mockSupervisors has 1 running supervisor
        expect(state.runningSupervisors).toBe(1);
        done();
      }
    });
  });

  it('should calculate running workflows correctly', (done) => {
    let emissionCount = 0;
    component.state$.subscribe(state => {
      emissionCount++;
      if (emissionCount === 2 && !state.loading) {
        // mockWorkflows has 0 running workflows
        expect(state.runningWorkflows).toBe(0);
        done();
      }
    });
  });

  it('should handle API errors gracefully', (done) => {
    apiService.getAgents.and.returnValue(
      throwError(() => ({ message: 'Failed to load data' }))
    );

    const newComponent = new DashboardComponent(apiService);

    newComponent.state$.subscribe(state => {
      if (state.error) {
        expect(state.loading).toBe(false);
        expect(state.error).toBe('Failed to load data');
        expect(state.agents).toEqual([]);
        expect(state.supervisors).toEqual([]);
        expect(state.workflows).toEqual([]);
        expect(state.runningAgents).toBe(0);
        expect(state.runningSupervisors).toBe(0);
        expect(state.runningWorkflows).toBe(0);
        done();
      }
    });
  });

  it('should use combineLatest to fetch all data in parallel', (done) => {
    let emissionCount = 0;
    component.state$.subscribe(state => {
      emissionCount++;
      if (emissionCount === 2 && !state.loading) {
        expect(apiService.getAgents).toHaveBeenCalled();
        expect(apiService.getSupervisors).toHaveBeenCalled();
        expect(apiService.getWorkflows).toHaveBeenCalled();
        done();
      }
    });
  });

  it('should compute derived state from combined data', (done) => {
    let emissionCount = 0;
    component.state$.subscribe(state => {
      emissionCount++;
      if (emissionCount === 2 && !state.loading) {
        expect(state.agents.length).toBe(2);
        expect(state.supervisors.length).toBe(2);
        expect(state.workflows.length).toBe(1);
        expect(state.runningAgents).toBe(1);
        expect(state.runningSupervisors).toBe(1);
        expect(state.runningWorkflows).toBe(0);
        done();
      }
    });
  });
});
