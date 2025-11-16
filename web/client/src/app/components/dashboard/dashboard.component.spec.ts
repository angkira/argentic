import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { RouterTestingModule } from '@angular/router/testing';
import { DashboardComponent } from './dashboard.component';
import { ApiService } from '../../services/api.service';
import { of } from 'rxjs';

describe('DashboardComponent', () => {
  let component: DashboardComponent;
  let fixture: ComponentFixture<DashboardComponent>;
  let apiService: jasmine.SpyObj<ApiService>;

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
    fixture = TestBed.createComponent(DashboardComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should load data on init', async () => {
    const mockAgents = [
      {
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
      }
    ];

    const mockSupervisors = [
      {
        id: '1',
        role: 'test_supervisor',
        description: 'Test',
        worker_agents: [],
        enable_dialogue_logging: true,
        max_dialogue_history_items: 100,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        status: 'stopped'
      }
    ];

    const mockWorkflows = [
      {
        id: '1',
        name: 'test_workflow',
        description: 'Test',
        nodes: [],
        edges: [],
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        status: 'stopped'
      }
    ];

    apiService.getAgents.and.returnValue(of(mockAgents));
    apiService.getSupervisors.and.returnValue(of(mockSupervisors));
    apiService.getWorkflows.and.returnValue(of(mockWorkflows));

    component.ngOnInit();
    await fixture.whenStable();

    expect(component.agents.length).toBe(1);
    expect(component.supervisors.length).toBe(1);
    expect(component.workflows.length).toBe(1);
    expect(component.loading).toBe(false);
  });

  it('should calculate running agents correctly', () => {
    component.agents = [
      {
        id: '1',
        role: 'agent1',
        description: 'Test',
        expected_output_format: 'json',
        enable_dialogue_logging: false,
        max_consecutive_tool_calls: 3,
        max_dialogue_history_items: 100,
        max_context_iterations: 10,
        enable_adaptive_context_management: true,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        status: 'running'
      },
      {
        id: '2',
        role: 'agent2',
        description: 'Test',
        expected_output_format: 'json',
        enable_dialogue_logging: false,
        max_consecutive_tool_calls: 3,
        max_dialogue_history_items: 100,
        max_context_iterations: 10,
        enable_adaptive_context_management: true,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        status: 'stopped'
      }
    ];

    expect(component.runningAgents).toBe(1);
  });
});
