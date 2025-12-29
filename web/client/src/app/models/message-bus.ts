export interface MessageBusMessage {
  id: string;
  timestamp: string;
  topic: string;
  agent_id: string;
  agent_role: string;
  message_type: 'request' | 'response' | 'event' | 'error';
  content: any;
  metadata?: Record<string, any>;
}

export interface MessageBusFilter {
  agentId?: string;
  agentRole?: string;
  topic?: string;
  messageType?: 'request' | 'response' | 'event' | 'error';
  searchTerm?: string;
}

export interface MessageBusStats {
  totalMessages: number;
  messagesByAgent: Record<string, number>;
  messagesByTopic: Record<string, number>;
  messagesByType: Record<string, number>;
}
