"""Unit tests for services."""

import pytest
from app.models import AgentCreate, AgentUpdate


class TestAgentService:
    """Test AgentService."""

    @pytest.mark.asyncio
    async def test_create_agent(self, agent_service, sample_agent_data):
        """Test creating an agent."""
        agent_create = AgentCreate(**sample_agent_data)
        agent = await agent_service.create_agent(agent_create)

        assert agent.id is not None
        assert agent.role == sample_agent_data["role"]
        assert agent.description == sample_agent_data["description"]
        assert agent.status == "stopped"
        assert agent.created_at is not None
        assert agent.updated_at is not None

    @pytest.mark.asyncio
    async def test_get_agent(self, agent_service, sample_agent_data):
        """Test getting an agent by ID."""
        agent_create = AgentCreate(**sample_agent_data)
        created_agent = await agent_service.create_agent(agent_create)

        retrieved_agent = await agent_service.get_agent(created_agent.id)

        assert retrieved_agent is not None
        assert retrieved_agent.id == created_agent.id
        assert retrieved_agent.role == created_agent.role

    @pytest.mark.asyncio
    async def test_get_nonexistent_agent(self, agent_service):
        """Test getting a non-existent agent."""
        agent = await agent_service.get_agent("nonexistent_id")
        assert agent is None

    @pytest.mark.asyncio
    async def test_list_agents(self, agent_service, sample_agent_data):
        """Test listing all agents."""
        # Create multiple agents
        agent1 = AgentCreate(**sample_agent_data)
        agent2_data = {**sample_agent_data, "role": "test_agent_2"}
        agent2 = AgentCreate(**agent2_data)

        await agent_service.create_agent(agent1)
        await agent_service.create_agent(agent2)

        agents = await agent_service.list_agents()

        assert len(agents) >= 2
        roles = [agent.role for agent in agents]
        assert "test_agent" in roles
        assert "test_agent_2" in roles

    @pytest.mark.asyncio
    async def test_update_agent(self, agent_service, sample_agent_data):
        """Test updating an agent."""
        agent_create = AgentCreate(**sample_agent_data)
        created_agent = await agent_service.create_agent(agent_create)

        update_data = AgentUpdate(description="Updated description")
        updated_agent = await agent_service.update_agent(created_agent.id, update_data)

        assert updated_agent is not None
        assert updated_agent.description == "Updated description"
        assert updated_agent.role == sample_agent_data["role"]  # Unchanged

    @pytest.mark.asyncio
    async def test_update_nonexistent_agent(self, agent_service):
        """Test updating a non-existent agent."""
        update_data = AgentUpdate(description="Updated description")
        result = await agent_service.update_agent("nonexistent_id", update_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_agent(self, agent_service, sample_agent_data):
        """Test deleting an agent."""
        agent_create = AgentCreate(**sample_agent_data)
        created_agent = await agent_service.create_agent(agent_create)

        deleted = await agent_service.delete_agent(created_agent.id)
        assert deleted is True

        # Verify agent is deleted
        agent = await agent_service.get_agent(created_agent.id)
        assert agent is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_agent(self, agent_service):
        """Test deleting a non-existent agent."""
        deleted = await agent_service.delete_agent("nonexistent_id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_update_agent_status(self, agent_service, sample_agent_data):
        """Test updating agent status."""
        agent_create = AgentCreate(**sample_agent_data)
        created_agent = await agent_service.create_agent(agent_create)

        updated_agent = await agent_service.update_agent_status(created_agent.id, "running")

        assert updated_agent is not None
        assert updated_agent.status == "running"


class TestSupervisorService:
    """Test SupervisorService."""

    @pytest.mark.asyncio
    async def test_create_supervisor(self, supervisor_service, sample_supervisor_data):
        """Test creating a supervisor."""
        from app.models import SupervisorCreate

        supervisor_create = SupervisorCreate(**sample_supervisor_data)
        supervisor = await supervisor_service.create_supervisor(supervisor_create)

        assert supervisor.id is not None
        assert supervisor.role == sample_supervisor_data["role"]
        assert len(supervisor.worker_agents) == 2
        assert supervisor.status == "stopped"

    @pytest.mark.asyncio
    async def test_list_supervisors(self, supervisor_service, sample_supervisor_data):
        """Test listing supervisors."""
        from app.models import SupervisorCreate

        supervisor_create = SupervisorCreate(**sample_supervisor_data)
        await supervisor_service.create_supervisor(supervisor_create)

        supervisors = await supervisor_service.list_supervisors()

        assert len(supervisors) >= 1
        assert any(s.role == "test_supervisor" for s in supervisors)


class TestWorkflowService:
    """Test WorkflowService."""

    @pytest.mark.asyncio
    async def test_create_workflow(self, workflow_service, sample_workflow_data):
        """Test creating a workflow."""
        from app.models import WorkflowCreate

        workflow_create = WorkflowCreate(**sample_workflow_data)
        workflow = await workflow_service.create_workflow(workflow_create)

        assert workflow.id is not None
        assert workflow.name == sample_workflow_data["name"]
        assert len(workflow.nodes) == 2
        assert len(workflow.edges) == 1
        assert workflow.status == "stopped"

    @pytest.mark.asyncio
    async def test_update_workflow(self, workflow_service, sample_workflow_data):
        """Test updating a workflow."""
        from app.models import WorkflowCreate, WorkflowUpdate

        workflow_create = WorkflowCreate(**sample_workflow_data)
        created_workflow = await workflow_service.create_workflow(workflow_create)

        update_data = WorkflowUpdate(name="Updated Workflow")
        updated_workflow = await workflow_service.update_workflow(
            created_workflow.id, update_data
        )

        assert updated_workflow is not None
        assert updated_workflow.name == "Updated Workflow"
        assert len(updated_workflow.nodes) == 2  # Unchanged
