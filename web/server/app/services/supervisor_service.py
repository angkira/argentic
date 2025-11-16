"""Service for managing supervisor configurations."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional
from app.models import SupervisorCreate, SupervisorUpdate, SupervisorResponse


class SupervisorService:
    """Service for managing supervisor configurations."""

    def __init__(self):
        """Initialize the supervisor service."""
        self._supervisors: Dict[str, Dict] = {}

    async def create_supervisor(self, supervisor_data: SupervisorCreate) -> SupervisorResponse:
        """Create a new supervisor configuration.

        Args:
            supervisor_data: Supervisor configuration data

        Returns:
            Created supervisor response
        """
        supervisor_id = str(uuid.uuid4())
        now = datetime.utcnow()

        supervisor = {
            "id": supervisor_id,
            "role": supervisor_data.role,
            "description": supervisor_data.description,
            "system_prompt": supervisor_data.system_prompt,
            "worker_agents": [agent.model_dump() for agent in supervisor_data.worker_agents],
            "enable_dialogue_logging": supervisor_data.enable_dialogue_logging,
            "max_dialogue_history_items": supervisor_data.max_dialogue_history_items,
            "llm_config_id": supervisor_data.llm_config_id,
            "created_at": now,
            "updated_at": now,
            "status": "stopped",
        }

        self._supervisors[supervisor_id] = supervisor
        return SupervisorResponse(**supervisor)

    async def get_supervisor(self, supervisor_id: str) -> Optional[SupervisorResponse]:
        """Get a supervisor by ID.

        Args:
            supervisor_id: Supervisor ID

        Returns:
            Supervisor response or None if not found
        """
        supervisor = self._supervisors.get(supervisor_id)
        if supervisor:
            return SupervisorResponse(**supervisor)
        return None

    async def list_supervisors(self) -> List[SupervisorResponse]:
        """List all supervisors.

        Returns:
            List of supervisor responses
        """
        return [SupervisorResponse(**supervisor) for supervisor in self._supervisors.values()]

    async def update_supervisor(
        self, supervisor_id: str, supervisor_data: SupervisorUpdate
    ) -> Optional[SupervisorResponse]:
        """Update a supervisor configuration.

        Args:
            supervisor_id: Supervisor ID
            supervisor_data: Supervisor update data

        Returns:
            Updated supervisor response or None if not found
        """
        supervisor = self._supervisors.get(supervisor_id)
        if not supervisor:
            return None

        # Update only provided fields
        update_dict = supervisor_data.model_dump(exclude_unset=True)
        if "worker_agents" in update_dict and update_dict["worker_agents"]:
            update_dict["worker_agents"] = [
                agent.model_dump() for agent in supervisor_data.worker_agents
            ]

        supervisor.update(update_dict)
        supervisor["updated_at"] = datetime.utcnow()

        return SupervisorResponse(**supervisor)

    async def delete_supervisor(self, supervisor_id: str) -> bool:
        """Delete a supervisor.

        Args:
            supervisor_id: Supervisor ID

        Returns:
            True if deleted, False if not found
        """
        if supervisor_id in self._supervisors:
            del self._supervisors[supervisor_id]
            return True
        return False

    async def update_supervisor_status(
        self, supervisor_id: str, status: str
    ) -> Optional[SupervisorResponse]:
        """Update supervisor status.

        Args:
            supervisor_id: Supervisor ID
            status: New status

        Returns:
            Updated supervisor response or None if not found
        """
        supervisor = self._supervisors.get(supervisor_id)
        if not supervisor:
            return None

        supervisor["status"] = status
        supervisor["updated_at"] = datetime.utcnow()

        return SupervisorResponse(**supervisor)
