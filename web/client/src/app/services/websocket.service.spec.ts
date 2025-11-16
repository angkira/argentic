import { TestBed } from '@angular/core/testing';
import { WebSocketService } from './websocket.service';
import { MessageBusMessage } from '../models';

// Mock Socket.IO
class MockSocket {
  private events: Map<string, Function[]> = new Map();
  connected = false;

  on(event: string, handler: Function) {
    if (!this.events.has(event)) {
      this.events.set(event, []);
    }
    this.events.get(event)!.push(handler);
  }

  emit(event: string, data?: any) {
    const handlers = this.events.get(event);
    if (handlers) {
      handlers.forEach(handler => handler(data));
    }
  }

  disconnect() {
    this.connected = false;
    this.emit('disconnect');
  }

  simulateConnect() {
    this.connected = true;
    this.emit('connect');
  }

  simulateError(error: Error) {
    this.emit('connect_error', error);
  }

  simulateMessage(message: MessageBusMessage) {
    this.emit('message_bus', message);
  }
}

describe('WebSocketService', () => {
  let service: WebSocketService;
  let mockSocket: MockSocket;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(WebSocketService);
    mockSocket = new MockSocket();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should emit initial disconnected state', (done) => {
    service.connectionState$.subscribe(state => {
      expect(state.status).toBe('disconnected');
      done();
    });
  });

  it('should have messages$ observable', (done) => {
    service.messages$.subscribe(messages => {
      expect(messages).toBeDefined();
      done();
    });
  });

  it('should filter messages by topic', (done) => {
    const testMessage: MessageBusMessage = {
      id: '1',
      timestamp: new Date().toISOString(),
      topic: 'test-topic',
      agent_id: 'agent-1',
      agent_role: 'test_agent',
      message_type: 'event',
      content: { test: 'data' }
    };

    service.messagesByTopic$('test-topic').subscribe(message => {
      expect(message.topic).toBe('test-topic');
      expect(message.id).toBe('1');
      done();
    });

    // Simulate receiving a message through the private subject
    // In real implementation, this would come through WebSocket
    (service as any).messageSubject$.next(testMessage);
  });

  it('should filter messages by agent', (done) => {
    const testMessage: MessageBusMessage = {
      id: '2',
      timestamp: new Date().toISOString(),
      topic: 'agent-topic',
      agent_id: 'agent-123',
      agent_role: 'research_agent',
      message_type: 'response',
      content: { result: 'success' }
    };

    service.messagesByAgent$('agent-123').subscribe(message => {
      expect(message.agent_id).toBe('agent-123');
      expect(message.id).toBe('2');
      done();
    });

    (service as any).messageSubject$.next(testMessage);
  });

  it('should filter messages by type', (done) => {
    const testMessage: MessageBusMessage = {
      id: '3',
      timestamp: new Date().toISOString(),
      topic: 'errors',
      agent_id: 'agent-1',
      agent_role: 'test_agent',
      message_type: 'error',
      content: { error: 'Something went wrong' }
    };

    service.messagesByType$('error').subscribe(message => {
      expect(message.message_type).toBe('error');
      expect(message.id).toBe('3');
      done();
    });

    (service as any).messageSubject$.next(testMessage);
  });

  it('should handle multiple messages in stream', (done) => {
    const messages: MessageBusMessage[] = [];

    service.messages$.subscribe(message => {
      messages.push(message);
      if (messages.length === 2) {
        expect(messages[0].id).toBe('msg-1');
        expect(messages[1].id).toBe('msg-2');
        done();
      }
    });

    (service as any).messageSubject$.next({
      id: 'msg-1',
      timestamp: new Date().toISOString(),
      topic: 'test',
      agent_id: 'agent-1',
      agent_role: 'test',
      message_type: 'event',
      content: {}
    });

    (service as any).messageSubject$.next({
      id: 'msg-2',
      timestamp: new Date().toISOString(),
      topic: 'test',
      agent_id: 'agent-2',
      agent_role: 'test',
      message_type: 'request',
      content: {}
    });
  });

  it('should share replay messages for multiple subscribers', (done) => {
    const message: MessageBusMessage = {
      id: 'shared-1',
      timestamp: new Date().toISOString(),
      topic: 'shared',
      agent_id: 'agent-1',
      agent_role: 'test',
      message_type: 'event',
      content: { shared: true }
    };

    // First subscriber
    service.messages$.subscribe(msg => {
      expect(msg.id).toBe('shared-1');
    });

    // Emit message
    (service as any).messageSubject$.next(message);

    // Second subscriber (should get replayed message)
    setTimeout(() => {
      service.messages$.subscribe(msg => {
        expect(msg.id).toBe('shared-1');
        done();
      });
    }, 50);
  });

  it('should provide factory methods for creating filtered streams', () => {
    expect(typeof service.messagesByTopic$).toBe('function');
    expect(typeof service.messagesByAgent$).toBe('function');
    expect(typeof service.messagesByType$).toBe('function');
  });

  it('should handle messages with metadata', (done) => {
    const messageWithMetadata: MessageBusMessage = {
      id: 'meta-1',
      timestamp: new Date().toISOString(),
      topic: 'metadata-test',
      agent_id: 'agent-1',
      agent_role: 'test',
      message_type: 'event',
      content: { data: 'test' },
      metadata: { source: 'unit-test', priority: 'high' }
    };

    service.messages$.subscribe(message => {
      expect(message.metadata).toBeDefined();
      expect(message.metadata?.['source']).toBe('unit-test');
      expect(message.metadata?.['priority']).toBe('high');
      done();
    });

    (service as any).messageSubject$.next(messageWithMetadata);
  });
});
