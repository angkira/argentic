export interface WorkerAgentConfig {
  role: string;
  description: string;
}

export interface SupervisorConfig {
  role: string;
  description: string;
  system_prompt?: string;
  worker_agents: WorkerAgentConfig[];
  enable_dialogue_logging: boolean;
  max_dialogue_history_items: number;
}

export interface SupervisorCreate extends SupervisorConfig {
  llm_config_id?: string;
}

export interface SupervisorUpdate {
  description?: string;
  system_prompt?: string;
  worker_agents?: WorkerAgentConfig[];
  enable_dialogue_logging?: boolean;
  max_dialogue_history_items?: number;
}

export interface Supervisor extends SupervisorConfig {
  id: string;
  created_at: string;
  updated_at: string;
  status: string;
}
