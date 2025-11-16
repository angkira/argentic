export type OutputFormat = 'json' | 'text' | 'code';
export type AgentStatus = 'stopped' | 'running' | 'error';

export interface AgentConfig {
  role: string;
  description: string;
  system_prompt?: string;
  expected_output_format: OutputFormat;
  enable_dialogue_logging: boolean;
  max_consecutive_tool_calls: number;
  max_dialogue_history_items: number;
  max_context_iterations: number;
  enable_adaptive_context_management: boolean;
}

export interface AgentCreate extends AgentConfig {
  llm_config_id?: string;
}

export interface AgentUpdate {
  description?: string;
  system_prompt?: string;
  expected_output_format?: OutputFormat;
  enable_dialogue_logging?: boolean;
  max_consecutive_tool_calls?: number;
  max_dialogue_history_items?: number;
}

export interface Agent extends AgentConfig {
  id: string;
  created_at: string;
  updated_at: string;
  status: AgentStatus;
}
