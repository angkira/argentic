"""Service for managing agent configurations."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional
from app.models import AgentCreate, AgentUpdate, AgentResponse


class AgentService:
    """Service for managing agent configurations."""

    def __init__(self):
        """Initialize the agent service."""
        self._agents: Dict[str, Dict] = {}

    async def create_agent(self, agent_data: AgentCreate) -> AgentResponse:
        """Create a new agent configuration.

        Args:
            agent_data: Agent configuration data

        Returns:
            Created agent response
        """
        agent_id = str(uuid.uuid4())
        now = datetime.utcnow()

        agent = {
            "id": agent_id,
            "role": agent_data.role,
            "description": agent_data.description,
            "system_prompt": agent_data.system_prompt,
            "expected_output_format": agent_data.expected_output_format,
            "enable_dialogue_logging": agent_data.enable_dialogue_logging,
            "max_consecutive_tool_calls": agent_data.max_consecutive_tool_calls,
            "max_dialogue_history_items": agent_data.max_dialogue_history_items,
            "max_context_iterations": agent_data.max_context_iterations,
            "enable_adaptive_context_management": agent_data.enable_adaptive_context_management,
            "llm_config_id": agent_data.llm_config_id,
            "created_at": now,
            "updated_at": now,
            "status": "stopped",
        }

        self._agents[agent_id] = agent
        return AgentResponse(**agent)

    async def get_agent(self, agent_id: str) -> Optional[AgentResponse]:
        """Get an agent by ID.

        Args:
            agent_id: Agent ID

        Returns:
            Agent response or None if not found
        """
        agent = self._agents.get(agent_id)
        if agent:
            return AgentResponse(**agent)
        return None

    async def list_agents(self) -> List[AgentResponse]:
        """List all agents.

        Returns:
            List of agent responses
        """
        return [AgentResponse(**agent) for agent in self._agents.values()]

    async def update_agent(self, agent_id: str, agent_data: AgentUpdate) -> Optional[AgentResponse]:
        """Update an agent configuration.

        Args:
            agent_id: Agent ID
            agent_data: Agent update data

        Returns:
            Updated agent response or None if not found
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return None

        # Update only provided fields
        update_dict = agent_data.model_dump(exclude_unset=True)
        agent.update(update_dict)
        agent["updated_at"] = datetime.utcnow()

        return AgentResponse(**agent)

    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent.

        Args:
            agent_id: Agent ID

        Returns:
            True if deleted, False if not found
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    async def update_agent_status(
        self, agent_id: str, status: str
    ) -> Optional[AgentResponse]:
        """Update agent status.

        Args:
            agent_id: Agent ID
            status: New status

        Returns:
            Updated agent response or None if not found
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return None

        agent["status"] = status
        agent["updated_at"] = datetime.utcnow()

        return AgentResponse(**agent)
