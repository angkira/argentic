export type LLMProviderType = 'ollama' | 'llama_cpp_server' | 'llama_cpp_cli' | 'google_gemini' | 'mock';
export type MessagingProtocol = 'mqtt' | 'rabbitmq' | 'kafka' | 'redis';

export interface LLMProviderConfig {
  provider: LLMProviderType;
  ollama_model_name?: string;
  ollama_base_url?: string;
  llama_cpp_server_host?: string;
  llama_cpp_server_port?: number;
  llama_cpp_server_auto_start?: boolean;
  llama_cpp_cli_binary?: string;
  llama_cpp_cli_model_path?: string;
  google_gemini_api_key?: string;
  google_gemini_model_name?: string;
  parameters?: Record<string, any>;
}

export interface MessagingConfig {
  protocol: MessagingProtocol;
  broker_address: string;
  port: number;
  client_id?: string;
  username?: string;
  password?: string;
  keepalive: number;
  use_tls: boolean;
}

export interface LLMProviderInfo {
  name: string;
  display_name: string;
  description: string;
  required_fields: string[];
}

export interface MessagingProtocolInfo {
  name: string;
  display_name: string;
  description: string;
  default_port: number;
}
