import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { ApiService } from './api.service';
import { Agent, Supervisor, Workflow } from '../models';

describe('ApiService', () => {
  let service: ApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [ApiService]
    });
    service = TestBed.inject(ApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  describe('Agent Operations', () => {
    it('should be created', () => {
      expect(service).toBeTruthy();
    });

    it('should get agents', () => {
      const mockAgents: Agent[] = [
        {
          id: '1',
          role: 'test_agent',
          description: 'Test agent',
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

      service.getAgents().subscribe(agents => {
        expect(agents).toEqual(mockAgents);
        expect(agents.length).toBe(1);
      });

      const req = httpMock.expectOne('/api/agents');
      expect(req.request.method).toBe('GET');
      req.flush(mockAgents);
    });

    it('should create agent', () => {
      const newAgent = {
        role: 'test_agent',
        description: 'Test agent',
        expected_output_format: 'json' as const,
        enable_dialogue_logging: false,
        max_consecutive_tool_calls: 3,
        max_dialogue_history_items: 100,
        max_context_iterations: 10,
        enable_adaptive_context_management: true
      };

      const mockResponse: Agent = {
        id: '1',
        ...newAgent,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        status: 'stopped'
      };

      service.createAgent(newAgent).subscribe(agent => {
        expect(agent).toEqual(mockResponse);
      });

      const req = httpMock.expectOne('/api/agents');
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(newAgent);
      req.flush(mockResponse);
    });

    it('should update agent', () => {
      const update = { description: 'Updated description' };
      const mockResponse: Agent = {
        id: '1',
        role: 'test_agent',
        description: 'Updated description',
        expected_output_format: 'json',
        enable_dialogue_logging: false,
        max_consecutive_tool_calls: 3,
        max_dialogue_history_items: 100,
        max_context_iterations: 10,
        enable_adaptive_context_management: true,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        status: 'stopped'
      };

      service.updateAgent('1', update).subscribe(agent => {
        expect(agent.description).toBe('Updated description');
      });

      const req = httpMock.expectOne('/api/agents/1');
      expect(req.request.method).toBe('PATCH');
      req.flush(mockResponse);
    });

    it('should delete agent', () => {
      service.deleteAgent('1').subscribe();

      const req = httpMock.expectOne('/api/agents/1');
      expect(req.request.method).toBe('DELETE');
      req.flush(null);
    });

    it('should start agent', () => {
      const mockResponse: Agent = {
        id: '1',
        role: 'test_agent',
        description: 'Test agent',
        expected_output_format: 'json',
        enable_dialogue_logging: false,
        max_consecutive_tool_calls: 3,
        max_dialogue_history_items: 100,
        max_context_iterations: 10,
        enable_adaptive_context_management: true,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        status: 'running'
      };

      service.startAgent('1').subscribe(agent => {
        expect(agent.status).toBe('running');
      });

      const req = httpMock.expectOne('/api/agents/1/start');
      expect(req.request.method).toBe('POST');
      req.flush(mockResponse);
    });
  });

  describe('Supervisor Operations', () => {
    it('should get supervisors', () => {
      const mockSupervisors: Supervisor[] = [
        {
          id: '1',
          role: 'test_supervisor',
          description: 'Test supervisor',
          worker_agents: [],
          enable_dialogue_logging: true,
          max_dialogue_history_items: 100,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
          status: 'stopped'
        }
      ];

      service.getSupervisors().subscribe(supervisors => {
        expect(supervisors).toEqual(mockSupervisors);
      });

      const req = httpMock.expectOne('/api/supervisors');
      expect(req.request.method).toBe('GET');
      req.flush(mockSupervisors);
    });
  });

  describe('Workflow Operations', () => {
    it('should get workflows', () => {
      const mockWorkflows: Workflow[] = [
        {
          id: '1',
          name: 'test_workflow',
          description: 'Test workflow',
          nodes: [],
          edges: [],
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
          status: 'stopped'
        }
      ];

      service.getWorkflows().subscribe(workflows => {
        expect(workflows).toEqual(mockWorkflows);
      });

      const req = httpMock.expectOne('/api/workflows');
      expect(req.request.method).toBe('GET');
      req.flush(mockWorkflows);
    });
  });

  describe('Configuration Operations', () => {
    it('should get LLM providers', () => {
      const mockProviders = [
        {
          name: 'google_gemini',
          display_name: 'Google Gemini',
          description: 'Cloud-based model',
          required_fields: ['google_gemini_api_key']
        }
      ];

      service.getLLMProviders().subscribe(providers => {
        expect(providers).toEqual(mockProviders);
      });

      const req = httpMock.expectOne('/api/config/llm-providers');
      expect(req.request.method).toBe('GET');
      req.flush(mockProviders);
    });

    it('should get messaging protocols', () => {
      const mockProtocols = [
        {
          name: 'mqtt',
          display_name: 'MQTT',
          description: 'Lightweight messaging',
          default_port: 1883
        }
      ];

      service.getMessagingProtocols().subscribe(protocols => {
        expect(protocols).toEqual(mockProtocols);
      });

      const req = httpMock.expectOne('/api/config/messaging-protocols');
      expect(req.request.method).toBe('GET');
      req.flush(mockProtocols);
    });
  });
});
