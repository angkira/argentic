"""Service for managing agent and workflow runtime execution."""

import asyncio
from typing import Dict, Optional, Any
from argentic import Agent, Messager, LLMFactory
from app.models import LLMProviderConfig, MessagingConfig


class RuntimeService:
    """Service for managing runtime execution of agents and workflows."""

    def __init__(self):
        """Initialize the runtime service."""
        self._running_agents: Dict[str, Agent] = {}
        self._running_supervisors: Dict[str, Any] = {}  # Type will be Supervisor when implemented
        self._messagers: Dict[str, Messager] = {}
        self._llm_providers: Dict[str, Any] = {}

    async def create_messager(self, config: MessagingConfig) -> Messager:
        """Create a messager instance from configuration.

        Args:
            config: Messaging configuration

        Returns:
            Messager instance
        """
        from argentic.core.messager.protocols import MessagerProtocol

        protocol_map = {
            "mqtt": MessagerProtocol.MQTT,
            "rabbitmq": MessagerProtocol.RABBITMQ,
            "kafka": MessagerProtocol.KAFKA,
            "redis": MessagerProtocol.REDIS,
        }

        messager_config = {
            "protocol": protocol_map[config.protocol],
            "broker_address": config.broker_address,
            "port": config.port,
            "keepalive": config.keepalive,
        }

        if config.client_id:
            messager_config["client_id"] = config.client_id
        if config.username:
            messager_config["username"] = config.username
        if config.password:
            messager_config["password"] = config.password
        if config.use_tls:
            messager_config["use_tls"] = config.use_tls

        messager = Messager(**messager_config)
        await messager.connect()

        return messager

    async def create_llm_provider(self, config: LLMProviderConfig) -> Any:
        """Create an LLM provider from configuration.

        Args:
            config: LLM provider configuration

        Returns:
            LLM provider instance
        """
        llm_config = {
            "provider": config.provider,
        }

        # Add provider-specific config
        if config.provider == "ollama":
            llm_config["ollama_model_name"] = config.ollama_model_name
            llm_config["ollama_base_url"] = config.ollama_base_url
        elif config.provider == "llama_cpp_server":
            llm_config["llama_cpp_server_host"] = config.llama_cpp_server_host
            llm_config["llama_cpp_server_port"] = config.llama_cpp_server_port
            llm_config["llama_cpp_server_auto_start"] = config.llama_cpp_server_auto_start
        elif config.provider == "llama_cpp_cli":
            llm_config["llama_cpp_cli_binary"] = config.llama_cpp_cli_binary
            llm_config["llama_cpp_cli_model_path"] = config.llama_cpp_cli_model_path
        elif config.provider == "google_gemini":
            llm_config["google_gemini_api_key"] = config.google_gemini_api_key
            llm_config["google_gemini_model_name"] = config.google_gemini_model_name

        # Add common parameters
        if config.parameters:
            llm_config.update({f"{config.provider}_parameters": config.parameters})

        llm = LLMFactory.create_llm(llm_config)
        await llm.start()

        return llm

    async def start_agent(
        self,
        agent_id: str,
        agent_config: Dict[str, Any],
        llm_config: LLMProviderConfig,
        messaging_config: MessagingConfig,
    ) -> None:
        """Start an agent.

        Args:
            agent_id: Agent ID
            agent_config: Agent configuration
            llm_config: LLM configuration
            messaging_config: Messaging configuration
        """
        if agent_id in self._running_agents:
            raise ValueError(f"Agent {agent_id} is already running")

        # Create messager
        messager = await self.create_messager(messaging_config)
        self._messagers[agent_id] = messager

        # Create LLM provider
        llm = await self.create_llm_provider(llm_config)
        self._llm_providers[agent_id] = llm

        # Create agent
        agent = Agent(
            llm=llm,
            messager=messager,
            role=agent_config["role"],
            description=agent_config["description"],
            system_prompt=agent_config.get("system_prompt"),
            expected_output_format=agent_config.get("expected_output_format", "json"),
            enable_dialogue_logging=agent_config.get("enable_dialogue_logging", False),
            max_consecutive_tool_calls=agent_config.get("max_consecutive_tool_calls", 3),
            max_dialogue_history_items=agent_config.get("max_dialogue_history_items", 100),
            max_context_iterations=agent_config.get("max_context_iterations", 10),
            enable_adaptive_context_management=agent_config.get(
                "enable_adaptive_context_management", True
            ),
        )

        await agent.async_init()
        self._running_agents[agent_id] = agent

    async def stop_agent(self, agent_id: str) -> None:
        """Stop a running agent.

        Args:
            agent_id: Agent ID
        """
        if agent_id not in self._running_agents:
            raise ValueError(f"Agent {agent_id} is not running")

        # Stop LLM provider
        if agent_id in self._llm_providers:
            await self._llm_providers[agent_id].stop()
            del self._llm_providers[agent_id]

        # Disconnect messager
        if agent_id in self._messagers:
            await self._messagers[agent_id].disconnect()
            del self._messagers[agent_id]

        # Remove agent
        del self._running_agents[agent_id]

    async def query_agent(self, agent_id: str, question: str) -> str:
        """Query a running agent.

        Args:
            agent_id: Agent ID
            question: Question to ask

        Returns:
            Agent response
        """
        if agent_id not in self._running_agents:
            raise ValueError(f"Agent {agent_id} is not running")

        agent = self._running_agents[agent_id]
        response = await agent.query(question)
        return response

    def is_agent_running(self, agent_id: str) -> bool:
        """Check if an agent is running.

        Args:
            agent_id: Agent ID

        Returns:
            True if running, False otherwise
        """
        return agent_id in self._running_agents

    async def cleanup(self) -> None:
        """Cleanup all running agents and resources."""
        agent_ids = list(self._running_agents.keys())
        for agent_id in agent_ids:
            try:
                await self.stop_agent(agent_id)
            except Exception as e:
                print(f"Error stopping agent {agent_id}: {e}")
