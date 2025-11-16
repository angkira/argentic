import { Component, OnInit, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { combineLatest, BehaviorSubject } from 'rxjs';
import { map, scan, startWith, shareReplay } from 'rxjs/operators';
import { WebSocketService } from '../../services/websocket.service';
import { ApiService } from '../../services/api.service';
import { MessageBusMessage, MessageBusFilter, Agent } from '../../models';

interface LogsState {
  messages: MessageBusMessage[];
  filteredMessages: MessageBusMessage[];
  agents: Agent[];
  topics: string[];
  loading: boolean;
  connected: boolean;
}

@Component({
  selector: 'app-logs',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './logs.component.html',
  styleUrls: ['./logs.component.scss']
})
export class LogsComponent implements OnInit {
  // Signals for local UI state
  selectedAgentId = signal<string>('');
  selectedTopic = signal<string>('');
  selectedMessageType = signal<MessageBusMessage['message_type'] | ''>('');
  searchTerm = signal<string>('');
  maxMessages = signal<number>(100);
  autoScroll = signal<boolean>(true);

  // Filter stream
  private filterSubject$ = new BehaviorSubject<MessageBusFilter>({});

  // Computed filter from signals
  private updateFilter = computed(() => {
    this.filterSubject$.next({
      agentId: this.selectedAgentId() || undefined,
      topic: this.selectedTopic() || undefined,
      messageType: this.selectedMessageType() || undefined,
      searchTerm: this.searchTerm() || undefined
    });
  });

  // Declarative streams
  private readonly connectionState$ = this.wsService.connectionState$;

  // Accumulate messages (keep last N messages)
  private readonly allMessages$ = this.wsService.messages$.pipe(
    scan((acc: MessageBusMessage[], msg: MessageBusMessage) => {
      const messages = [...acc, msg];
      return messages.slice(-this.maxMessages());
    }, []),
    startWith([] as MessageBusMessage[]),
    shareReplay({ bufferSize: 1, refCount: true }),
    takeUntilDestroyed()
  );

  // Get agents list
  private readonly agents$ = this.apiService.getAgents().pipe(
    startWith([]),
    shareReplay({ bufferSize: 1, refCount: true }),
    takeUntilDestroyed()
  );

  // Extract unique topics from messages
  private readonly topics$ = this.allMessages$.pipe(
    map(messages => {
      const topicSet = new Set(messages.map(m => m.topic));
      return Array.from(topicSet).sort();
    }),
    startWith([]),
    takeUntilDestroyed()
  );

  // Main state stream with filtering
  readonly state$ = combineLatest({
    messages: this.allMessages$,
    filter: this.filterSubject$,
    agents: this.agents$,
    topics: this.topics$,
    connectionState: this.connectionState$
  }).pipe(
    map(({ messages, filter, agents, topics, connectionState }) => {
      // Apply filters
      let filtered = messages;

      if (filter.agentId) {
        filtered = filtered.filter(m => m.agent_id === filter.agentId);
      }

      if (filter.topic) {
        filtered = filtered.filter(m => m.topic === filter.topic);
      }

      if (filter.messageType) {
        filtered = filtered.filter(m => m.message_type === filter.messageType);
      }

      if (filter.searchTerm) {
        const term = filter.searchTerm.toLowerCase();
        filtered = filtered.filter(m =>
          m.agent_role.toLowerCase().includes(term) ||
          m.topic.toLowerCase().includes(term) ||
          JSON.stringify(m.content).toLowerCase().includes(term)
        );
      }

      return {
        messages,
        filteredMessages: filtered,
        agents,
        topics,
        loading: false,
        connected: connectionState.status === 'connected'
      } as LogsState;
    }),
    startWith({
      messages: [],
      filteredMessages: [],
      agents: [],
      topics: [],
      loading: true,
      connected: false
    } as LogsState),
    shareReplay({ bufferSize: 1, refCount: true }),
    takeUntilDestroyed()
  );

  constructor(
    private wsService: WebSocketService,
    private apiService: ApiService
  ) {
    // Trigger filter update when signals change
    this.updateFilter;
  }

  ngOnInit(): void {
    // Connect to WebSocket
    this.wsService.connect();
  }

  clearFilters(): void {
    this.selectedAgentId.set('');
    this.selectedTopic.set('');
    this.selectedMessageType.set('');
    this.searchTerm.set('');
  }

  clearMessages(): void {
    // This would require a method to clear the accumulated messages
    // For now, we can just reload the page or implement a clear mechanism
    window.location.reload();
  }

  getMessageTypeClass(type: string): string {
    switch (type) {
      case 'request': return 'badge-secondary';
      case 'response': return 'badge-success';
      case 'event': return 'badge-primary';
      case 'error': return 'badge-danger';
      default: return 'badge-secondary';
    }
  }

  formatTimestamp(timestamp: string): string {
    return new Date(timestamp).toLocaleTimeString();
  }

  formatContent(content: any): string {
    if (typeof content === 'string') {
      return content;
    }
    return JSON.stringify(content, null, 2);
  }

  hasMetadata(metadata: any): boolean {
    return metadata && Object.keys(metadata).length > 0;
  }

  toggleAutoScroll(): void {
    this.autoScroll.set(!this.autoScroll());
  }
}
