import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { FormsModule } from '@angular/forms';
import { LogsComponent } from './logs.component';
import { WebSocketService } from '../../services/websocket.service';
import { ApiService } from '../../services/api.service';
import { of, Subject } from 'rxjs';
import { MessageBusMessage, Agent } from '../../models';

describe('LogsComponent', () => {
  let component: LogsComponent;
  let fixture: ComponentFixture<LogsComponent>;
  let wsService: jasmine.SpyObj<WebSocketService>;
  let apiService: jasmine.SpyObj<ApiService>;
  let messagesSubject: Subject<MessageBusMessage>;
  let connectionStateSubject: Subject<any>;

  const mockAgent: Agent = {
    id: 'agent-1',
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
  };

  const mockMessage: MessageBusMessage = {
    id: 'msg-1',
    timestamp: '2024-01-01T12:00:00Z',
    topic: 'test-topic',
    agent_id: 'agent-1',
    agent_role: 'test_agent',
    message_type: 'event',
    content: { test: 'data' },
    metadata: { source: 'test' }
  };

  beforeEach(async () => {
    messagesSubject = new Subject();
    connectionStateSubject = new Subject();

    const wsServiceSpy = jasmine.createSpyObj('WebSocketService', ['connect', 'disconnect', 'subscribeTopic', 'unsubscribeTopic']);
    const apiServiceSpy = jasmine.createSpyObj('ApiService', ['getAgents']);

    // Mock observable properties
    Object.defineProperty(wsServiceSpy, 'messages$', {
      get: () => messagesSubject.asObservable()
    });
    Object.defineProperty(wsServiceSpy, 'connectionState$', {
      get: () => connectionStateSubject.asObservable()
    });

    await TestBed.configureTestingModule({
      imports: [LogsComponent, HttpClientTestingModule, FormsModule],
      providers: [
        { provide: WebSocketService, useValue: wsServiceSpy },
        { provide: ApiService, useValue: apiServiceSpy }
      ]
    }).compileComponents();

    wsService = TestBed.inject(WebSocketService) as jasmine.SpyObj<WebSocketService>;
    apiService = TestBed.inject(ApiService) as jasmine.SpyObj<ApiService>;

    apiService.getAgents.and.returnValue(of([mockAgent]));

    fixture = TestBed.createComponent(LogsComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should connect to WebSocket on init', () => {
    component.ngOnInit();
    expect(wsService.connect).toHaveBeenCalled();
  });

  it('should initialize with default filter values', () => {
    expect(component.selectedAgentId()).toBe('');
    expect(component.selectedTopic()).toBe('');
    expect(component.selectedMessageType()).toBe('');
    expect(component.searchTerm()).toBe('');
  });

  it('should initialize with default UI state', () => {
    expect(component.maxMessages()).toBe(100);
    expect(component.autoScroll()).toBe(true);
  });

  it('should provide state$ observable', (done) => {
    connectionStateSubject.next({ status: 'connected' });

    component.state$.subscribe(state => {
      expect(state).toBeDefined();
      expect(state.loading).toBeDefined();
      expect(state.connected).toBeDefined();
      done();
    });
  });

  it('should toggle auto-scroll', () => {
    expect(component.autoScroll()).toBe(true);
    component.toggleAutoScroll();
    expect(component.autoScroll()).toBe(false);
    component.toggleAutoScroll();
    expect(component.autoScroll()).toBe(true);
  });

  it('should clear all filters', () => {
    component.selectedAgentId.set('agent-1');
    component.selectedTopic.set('test-topic');
    component.selectedMessageType.set('event');
    component.searchTerm.set('test');

    component.clearFilters();

    expect(component.selectedAgentId()).toBe('');
    expect(component.selectedTopic()).toBe('');
    expect(component.selectedMessageType()).toBe('');
    expect(component.searchTerm()).toBe('');
  });

  it('should get correct message type class', () => {
    expect(component.getMessageTypeClass('request')).toBe('badge-secondary');
    expect(component.getMessageTypeClass('response')).toBe('badge-success');
    expect(component.getMessageTypeClass('event')).toBe('badge-primary');
    expect(component.getMessageTypeClass('error')).toBe('badge-danger');
    expect(component.getMessageTypeClass('unknown')).toBe('badge-secondary');
  });

  it('should format timestamp correctly', () => {
    const timestamp = '2024-01-01T12:30:45Z';
    const formatted = component.formatTimestamp(timestamp);
    expect(formatted).toContain(':');
    expect(typeof formatted).toBe('string');
  });

  it('should format string content as-is', () => {
    const content = 'Simple string';
    expect(component.formatContent(content)).toBe('Simple string');
  });

  it('should format object content as JSON', () => {
    const content = { key: 'value', nested: { prop: 123 } };
    const formatted = component.formatContent(content);
    expect(formatted).toContain('key');
    expect(formatted).toContain('value');
    expect(formatted).toContain('nested');
  });

  it('should detect metadata presence', () => {
    expect(component.hasMetadata(null)).toBe(false);
    expect(component.hasMetadata(undefined)).toBe(false);
    expect(component.hasMetadata({})).toBe(false);
    expect(component.hasMetadata({ key: 'value' })).toBe(true);
  });

  it('should filter messages by agent ID', (done) => {
    component.selectedAgentId.set('agent-1');
    connectionStateSubject.next({ status: 'connected' });

    const message1 = { ...mockMessage, id: 'msg-1', agent_id: 'agent-1' };
    const message2 = { ...mockMessage, id: 'msg-2', agent_id: 'agent-2' };

    // Simulate messages
    setTimeout(() => {
      messagesSubject.next(message1);
      messagesSubject.next(message2);
    }, 50);

    setTimeout(() => {
      component.state$.subscribe(state => {
        const filtered = state.filteredMessages;
        if (filtered.length > 0) {
          expect(filtered.every(m => m.agent_id === 'agent-1')).toBe(true);
          done();
        }
      });
    }, 100);
  });

  it('should filter messages by topic', (done) => {
    component.selectedTopic.set('important');
    connectionStateSubject.next({ status: 'connected' });

    const message1 = { ...mockMessage, id: 'msg-1', topic: 'important' };
    const message2 = { ...mockMessage, id: 'msg-2', topic: 'other' };

    setTimeout(() => {
      messagesSubject.next(message1);
      messagesSubject.next(message2);
    }, 50);

    setTimeout(() => {
      component.state$.subscribe(state => {
        const filtered = state.filteredMessages;
        if (filtered.length > 0) {
          expect(filtered.every(m => m.topic === 'important')).toBe(true);
          done();
        }
      });
    }, 100);
  });

  it('should filter messages by message type', (done) => {
    component.selectedMessageType.set('error');
    connectionStateSubject.next({ status: 'connected' });

    const message1 = { ...mockMessage, id: 'msg-1', message_type: 'error' as const };
    const message2 = { ...mockMessage, id: 'msg-2', message_type: 'event' as const };

    setTimeout(() => {
      messagesSubject.next(message1);
      messagesSubject.next(message2);
    }, 50);

    setTimeout(() => {
      component.state$.subscribe(state => {
        const filtered = state.filteredMessages;
        if (filtered.length > 0) {
          expect(filtered.every(m => m.message_type === 'error')).toBe(true);
          done();
        }
      });
    }, 100);
  });

  it('should search messages by content', (done) => {
    component.searchTerm.set('special');
    connectionStateSubject.next({ status: 'connected' });

    const message1 = { ...mockMessage, id: 'msg-1', content: { data: 'special keyword' } };
    const message2 = { ...mockMessage, id: 'msg-2', content: { data: 'normal' } };

    setTimeout(() => {
      messagesSubject.next(message1);
      messagesSubject.next(message2);
    }, 50);

    setTimeout(() => {
      component.state$.subscribe(state => {
        const filtered = state.filteredMessages;
        if (filtered.length > 0) {
          const hasSearchTerm = filtered.some(m =>
            JSON.stringify(m.content).toLowerCase().includes('special')
          );
          expect(hasSearchTerm).toBe(true);
          done();
        }
      });
    }, 100);
  });

  it('should indicate connection status in state', (done) => {
    connectionStateSubject.next({ status: 'connected' });

    component.state$.subscribe(state => {
      if (state.connected) {
        expect(state.connected).toBe(true);
        done();
      }
    });
  });

  it('should reload page when clearing messages', () => {
    spyOn(window.location, 'reload');
    component.clearMessages();
    expect(window.location.reload).toHaveBeenCalled();
  });
});
