import { Injectable, inject } from '@angular/core';
import { Observable, Subject, fromEvent, merge } from 'rxjs';
import { map, filter, shareReplay, startWith, tap } from 'rxjs/operators';
import { io, Socket } from 'socket.io-client';
import { MessageBusMessage } from '../models';

export type ConnectionStatus = 'connected' | 'disconnected' | 'connecting' | 'error';

interface ConnectionState {
  status: ConnectionStatus;
  error?: string;
}

@Injectable({
  providedIn: 'root'
})
export class WebSocketService {
  private socket: Socket | null = null;
  private messageSubject$ = new Subject<MessageBusMessage>();

  // Connection state stream
  private connectionStateSubject$ = new Subject<ConnectionState>();
  readonly connectionState$ = this.connectionStateSubject$.pipe(
    startWith({ status: 'disconnected' as ConnectionStatus }),
    shareReplay({ bufferSize: 1, refCount: true })
  );

  // All messages stream (declarative)
  readonly messages$ = this.messageSubject$.pipe(
    shareReplay({ bufferSize: 100, refCount: true })
  );

  // Messages by topic (declarative factory method)
  messagesByTopic$(topic: string): Observable<MessageBusMessage> {
    return this.messages$.pipe(
      filter(msg => msg.topic === topic)
    );
  }

  // Messages by agent (declarative factory method)
  messagesByAgent$(agentId: string): Observable<MessageBusMessage> {
    return this.messages$.pipe(
      filter(msg => msg.agent_id === agentId)
    );
  }

  // Messages by type (declarative factory method)
  messagesByType$(messageType: MessageBusMessage['message_type']): Observable<MessageBusMessage> {
    return this.messages$.pipe(
      filter(msg => msg.message_type === messageType)
    );
  }

  connect(url: string = 'http://localhost:8000'): void {
    if (this.socket?.connected) {
      return;
    }

    this.connectionStateSubject$.next({ status: 'connecting' });

    this.socket = io(url, {
      path: '/ws/socket.io',
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    });

    // Setup event listeners
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.connectionStateSubject$.next({ status: 'connected' });
    });

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      this.connectionStateSubject$.next({ status: 'disconnected' });
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.connectionStateSubject$.next({
        status: 'error',
        error: error.message
      });
    });

    // Listen for message bus messages
    this.socket.on('message_bus', (message: MessageBusMessage) => {
      this.messageSubject$.next(message);
    });
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.connectionStateSubject$.next({ status: 'disconnected' });
    }
  }

  // Subscribe to specific topics
  subscribeTopic(topic: string): void {
    if (this.socket?.connected) {
      this.socket.emit('subscribe', { topic });
    }
  }

  // Unsubscribe from topics
  unsubscribeTopic(topic: string): void {
    if (this.socket?.connected) {
      this.socket.emit('unsubscribe', { topic });
    }
  }

  // Send message to a topic
  sendMessage(topic: string, message: any): void {
    if (this.socket?.connected) {
      this.socket.emit('publish', { topic, message });
    }
  }
}
