import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { RouterTestingModule } from '@angular/router/testing';
import { WorkflowsComponent } from './workflows.component';
import { ApiService } from '../../services/api.service';
import { of, throwError } from 'rxjs';
import { Workflow } from '../../models';

describe('WorkflowsComponent', () => {
  let component: WorkflowsComponent;
  let fixture: ComponentFixture<WorkflowsComponent>;
  let apiService: jasmine.SpyObj<ApiService>;

  const mockWorkflow: Workflow = {
    id: 'workflow-1',
    name: 'Test Workflow',
    description: 'Test workflow description',
    nodes: [],
    edges: [],
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    status: 'stopped'
  };

  beforeEach(async () => {
    const apiServiceSpy = jasmine.createSpyObj('ApiService', [
      'getWorkflows',
      'deleteWorkflow',
      'startWorkflow',
      'stopWorkflow'
    ]);

    await TestBed.configureTestingModule({
      imports: [WorkflowsComponent, HttpClientTestingModule, RouterTestingModule],
      providers: [
        { provide: ApiService, useValue: apiServiceSpy }
      ]
    }).compileComponents();

    apiService = TestBed.inject(ApiService) as jasmine.SpyObj<ApiService>;
    apiService.getWorkflows.and.returnValue(of([mockWorkflow]));

    fixture = TestBed.createComponent(WorkflowsComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should have state$ observable that emits initial loading state', (done) => {
    component.state$.subscribe(state => {
      if (state.loading) {
        expect(state.workflows).toEqual([]);
        expect(state.loading).toBe(true);
        expect(state.error).toBeNull();
        done();
      }
    });
  });

  it('should emit workflows data after loading', (done) => {
    let emissionCount = 0;
    component.state$.subscribe(state => {
      emissionCount++;
      if (emissionCount === 2 && !state.loading) {
        expect(state.workflows).toEqual([mockWorkflow]);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
        done();
      }
    });
  });

  it('should delete workflow after confirmation', (done) => {
    spyOn(window, 'confirm').and.returnValue(true);
    apiService.deleteWorkflow.and.returnValue(of(void 0));
    apiService.getWorkflows.and.returnValue(of([]));

    component.deleteWorkflow('workflow-1');

    setTimeout(() => {
      expect(apiService.deleteWorkflow).toHaveBeenCalledWith('workflow-1');
      expect(apiService.getWorkflows).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should not delete workflow if not confirmed', () => {
    spyOn(window, 'confirm').and.returnValue(false);

    component.deleteWorkflow('workflow-1');

    expect(apiService.deleteWorkflow).not.toHaveBeenCalled();
  });

  it('should handle delete workflow error', (done) => {
    spyOn(window, 'confirm').and.returnValue(true);
    spyOn(console, 'error');

    apiService.deleteWorkflow.and.returnValue(
      throwError(() => new Error('Delete failed'))
    );

    component.deleteWorkflow('workflow-1');

    setTimeout(() => {
      expect(console.error).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should start stopped workflow', (done) => {
    const stoppedWorkflow = { ...mockWorkflow, status: 'stopped' as const };
    apiService.startWorkflow.and.returnValue(of({ ...stoppedWorkflow, status: 'running' as const }));
    apiService.getWorkflows.and.returnValue(of([{ ...stoppedWorkflow, status: 'running' as const }]));

    component.toggleWorkflow(stoppedWorkflow);

    setTimeout(() => {
      expect(apiService.startWorkflow).toHaveBeenCalledWith('workflow-1');
      expect(apiService.getWorkflows).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should stop running workflow', (done) => {
    const runningWorkflow = { ...mockWorkflow, status: 'running' as const };
    apiService.stopWorkflow.and.returnValue(of({ ...runningWorkflow, status: 'stopped' as const }));
    apiService.getWorkflows.and.returnValue(of([{ ...runningWorkflow, status: 'stopped' as const }]));

    component.toggleWorkflow(runningWorkflow);

    setTimeout(() => {
      expect(apiService.stopWorkflow).toHaveBeenCalledWith('workflow-1');
      expect(apiService.getWorkflows).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should handle toggle workflow error', (done) => {
    spyOn(console, 'error');

    apiService.startWorkflow.and.returnValue(
      throwError(() => new Error('Toggle failed'))
    );

    component.toggleWorkflow(mockWorkflow);

    setTimeout(() => {
      expect(console.error).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should get correct status class for running', () => {
    expect(component.getStatusClass('running')).toBe('badge-success');
  });

  it('should get correct status class for stopped', () => {
    expect(component.getStatusClass('stopped')).toBe('badge-secondary');
  });

  it('should get correct status class for error', () => {
    expect(component.getStatusClass('error')).toBe('badge-danger');
  });

  it('should get default class for unknown status', () => {
    expect(component.getStatusClass('unknown')).toBe('badge-secondary');
  });

  it('should handle API error when loading workflows', (done) => {
    apiService.getWorkflows.and.returnValue(
      throwError(() => ({ message: 'Failed to load workflows' }))
    );

    const newComponent = new WorkflowsComponent(apiService);

    newComponent.state$.subscribe(state => {
      if (state.error) {
        expect(state.loading).toBe(false);
        expect(state.error).toBe('Failed to load workflows');
        expect(state.workflows).toEqual([]);
        done();
      }
    });
  });

  it('should refresh on initial load', () => {
    expect(apiService.getWorkflows).toHaveBeenCalled();
  });
});
