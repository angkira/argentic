"""Service for managing workflow configurations."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional
from app.models import WorkflowCreate, WorkflowUpdate, WorkflowResponse


class WorkflowService:
    """Service for managing workflow configurations."""

    def __init__(self):
        """Initialize the workflow service."""
        self._workflows: Dict[str, Dict] = {}

    async def create_workflow(self, workflow_data: WorkflowCreate) -> WorkflowResponse:
        """Create a new workflow configuration.

        Args:
            workflow_data: Workflow configuration data

        Returns:
            Created workflow response
        """
        workflow_id = str(uuid.uuid4())
        now = datetime.utcnow()

        workflow = {
            "id": workflow_id,
            "name": workflow_data.name,
            "description": workflow_data.description,
            "nodes": [node.model_dump() for node in workflow_data.nodes],
            "edges": [edge.model_dump() for edge in workflow_data.edges],
            "created_at": now,
            "updated_at": now,
            "status": "stopped",
        }

        self._workflows[workflow_id] = workflow
        return WorkflowResponse(**workflow)

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowResponse]:
        """Get a workflow by ID.

        Args:
            workflow_id: Workflow ID

        Returns:
            Workflow response or None if not found
        """
        workflow = self._workflows.get(workflow_id)
        if workflow:
            return WorkflowResponse(**workflow)
        return None

    async def list_workflows(self) -> List[WorkflowResponse]:
        """List all workflows.

        Returns:
            List of workflow responses
        """
        return [WorkflowResponse(**workflow) for workflow in self._workflows.values()]

    async def update_workflow(
        self, workflow_id: str, workflow_data: WorkflowUpdate
    ) -> Optional[WorkflowResponse]:
        """Update a workflow configuration.

        Args:
            workflow_id: Workflow ID
            workflow_data: Workflow update data

        Returns:
            Updated workflow response or None if not found
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None

        # Update only provided fields
        update_dict = workflow_data.model_dump(exclude_unset=True)
        if "nodes" in update_dict and update_dict["nodes"]:
            update_dict["nodes"] = [node.model_dump() for node in workflow_data.nodes]
        if "edges" in update_dict and update_dict["edges"]:
            update_dict["edges"] = [edge.model_dump() for edge in workflow_data.edges]

        workflow.update(update_dict)
        workflow["updated_at"] = datetime.utcnow()

        return WorkflowResponse(**workflow)

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            True if deleted, False if not found
        """
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            return True
        return False

    async def update_workflow_status(
        self, workflow_id: str, status: str
    ) -> Optional[WorkflowResponse]:
        """Update workflow status.

        Args:
            workflow_id: Workflow ID
            status: New status

        Returns:
            Updated workflow response or None if not found
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None

        workflow["status"] = status
        workflow["updated_at"] = datetime.utcnow()

        return WorkflowResponse(**workflow)
