import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { FormsModule } from '@angular/forms';
import { SupervisorsComponent } from './supervisors.component';
import { ApiService } from '../../services/api.service';
import { of, throwError } from 'rxjs';
import { Supervisor, SupervisorCreate } from '../../models';

describe('SupervisorsComponent', () => {
  let component: SupervisorsComponent;
  let fixture: ComponentFixture<SupervisorsComponent>;
  let apiService: jasmine.SpyObj<ApiService>;

  const mockSupervisor: Supervisor = {
    id: 'supervisor-1',
    role: 'test_supervisor',
    description: 'Test supervisor',
    system_prompt: 'Test prompt',
    worker_agents: ['agent-1', 'agent-2'],
    enable_dialogue_logging: true,
    max_dialogue_history_items: 100,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    status: 'stopped'
  };

  beforeEach(async () => {
    const apiServiceSpy = jasmine.createSpyObj('ApiService', [
      'getSupervisors',
      'createSupervisor',
      'deleteSupervisor',
      'startSupervisor',
      'stopSupervisor'
    ]);

    await TestBed.configureTestingModule({
      imports: [SupervisorsComponent, HttpClientTestingModule, FormsModule],
      providers: [
        { provide: ApiService, useValue: apiServiceSpy }
      ]
    }).compileComponents();

    apiService = TestBed.inject(ApiService) as jasmine.SpyObj<ApiService>;
    apiService.getSupervisors.and.returnValue(of([mockSupervisor]));

    fixture = TestBed.createComponent(SupervisorsComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should have state$ observable that emits initial loading state', (done) => {
    component.state$.subscribe(state => {
      if (state.loading) {
        expect(state.supervisors).toEqual([]);
        expect(state.loading).toBe(true);
        expect(state.error).toBeNull();
        done();
      }
    });
  });

  it('should emit supervisors data after loading', (done) => {
    let emissionCount = 0;
    component.state$.subscribe(state => {
      emissionCount++;
      if (emissionCount === 2 && !state.loading) {
        expect(state.supervisors).toEqual([mockSupervisor]);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
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
    component.newSupervisor.role = 'test_role';
    component.newSupervisor.description = 'test description';
    component.closeCreateModal();

    expect(component.newSupervisor.role).toBe('');
    expect(component.newSupervisor.description).toBe('');
  });

  it('should create supervisor and refresh list', (done) => {
    const createdSupervisor = { ...mockSupervisor, id: '2', role: 'new_supervisor' };
    apiService.createSupervisor.and.returnValue(of(createdSupervisor));
    apiService.getSupervisors.and.returnValue(of([createdSupervisor]));

    component.newSupervisor.role = 'new_supervisor';
    component.newSupervisor.description = 'New supervisor';
    component.createSupervisor();

    expect(component.showCreateModal()).toBe(false);

    setTimeout(() => {
      expect(apiService.getSupervisors).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should handle create supervisor error', (done) => {
    spyOn(console, 'error');
    spyOn(window, 'alert');

    apiService.createSupervisor.and.returnValue(
      throwError(() => new Error('Create failed'))
    );

    component.newSupervisor.role = 'test_supervisor';
    component.newSupervisor.description = 'Test description';
    component.createSupervisor();

    setTimeout(() => {
      expect(console.error).toHaveBeenCalled();
      expect(window.alert).toHaveBeenCalledWith('Failed to create supervisor');
      done();
    }, 100);
  });

  it('should delete supervisor after confirmation', (done) => {
    spyOn(window, 'confirm').and.returnValue(true);
    apiService.deleteSupervisor.and.returnValue(of(void 0));
    apiService.getSupervisors.and.returnValue(of([]));

    component.deleteSupervisor('supervisor-1');

    setTimeout(() => {
      expect(apiService.deleteSupervisor).toHaveBeenCalledWith('supervisor-1');
      expect(apiService.getSupervisors).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should not delete supervisor if not confirmed', () => {
    spyOn(window, 'confirm').and.returnValue(false);

    component.deleteSupervisor('supervisor-1');

    expect(apiService.deleteSupervisor).not.toHaveBeenCalled();
  });

  it('should handle delete supervisor error', (done) => {
    spyOn(window, 'confirm').and.returnValue(true);
    spyOn(console, 'error');

    apiService.deleteSupervisor.and.returnValue(
      throwError(() => new Error('Delete failed'))
    );

    component.deleteSupervisor('supervisor-1');

    setTimeout(() => {
      expect(console.error).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should start stopped supervisor', (done) => {
    const stoppedSupervisor = { ...mockSupervisor, status: 'stopped' as const };
    apiService.startSupervisor.and.returnValue(of({ ...stoppedSupervisor, status: 'running' as const }));
    apiService.getSupervisors.and.returnValue(of([{ ...stoppedSupervisor, status: 'running' as const }]));

    component.toggleSupervisor(stoppedSupervisor);

    setTimeout(() => {
      expect(apiService.startSupervisor).toHaveBeenCalledWith('supervisor-1');
      expect(apiService.getSupervisors).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should stop running supervisor', (done) => {
    const runningSupervisor = { ...mockSupervisor, status: 'running' as const };
    apiService.stopSupervisor.and.returnValue(of({ ...runningSupervisor, status: 'stopped' as const }));
    apiService.getSupervisors.and.returnValue(of([{ ...runningSupervisor, status: 'stopped' as const }]));

    component.toggleSupervisor(runningSupervisor);

    setTimeout(() => {
      expect(apiService.stopSupervisor).toHaveBeenCalledWith('supervisor-1');
      expect(apiService.getSupervisors).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should handle toggle supervisor error', (done) => {
    spyOn(console, 'error');

    apiService.startSupervisor.and.returnValue(
      throwError(() => new Error('Toggle failed'))
    );

    component.toggleSupervisor(mockSupervisor);

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

  it('should handle API error when loading supervisors', (done) => {
    apiService.getSupervisors.and.returnValue(
      throwError(() => ({ message: 'Failed to load supervisors' }))
    );

    const newComponent = new SupervisorsComponent(apiService);

    newComponent.state$.subscribe(state => {
      if (state.error) {
        expect(state.loading).toBe(false);
        expect(state.error).toBe('Failed to load supervisors');
        expect(state.supervisors).toEqual([]);
        done();
      }
    });
  });

  it('should use default supervisor values', () => {
    expect(component.defaultSupervisor.role).toBe('');
    expect(component.defaultSupervisor.description).toBe('');
    expect(component.defaultSupervisor.system_prompt).toBe('');
    expect(component.defaultSupervisor.worker_agents).toEqual([]);
    expect(component.defaultSupervisor.enable_dialogue_logging).toBe(true);
    expect(component.defaultSupervisor.max_dialogue_history_items).toBe(100);
  });
});
