"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services import AgentService, SupervisorService, WorkflowService


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def agent_service():
    """Create a fresh AgentService instance."""
    return AgentService()


@pytest.fixture
def supervisor_service():
    """Create a fresh SupervisorService instance."""
    return SupervisorService()


@pytest.fixture
def workflow_service():
    """Create a fresh WorkflowService instance."""
    return WorkflowService()


@pytest.fixture
def sample_agent_data():
    """Sample agent creation data."""
    return {
        "role": "test_agent",
        "description": "A test agent",
        "system_prompt": "You are a test agent",
        "expected_output_format": "json",
        "enable_dialogue_logging": False,
        "max_consecutive_tool_calls": 3,
        "max_dialogue_history_items": 100,
        "max_context_iterations": 10,
        "enable_adaptive_context_management": True,
    }


@pytest.fixture
def sample_supervisor_data():
    """Sample supervisor creation data."""
    return {
        "role": "test_supervisor",
        "description": "A test supervisor",
        "system_prompt": "You are a test supervisor",
        "worker_agents": [
            {"role": "worker1", "description": "First worker"},
            {"role": "worker2", "description": "Second worker"},
        ],
        "enable_dialogue_logging": True,
        "max_dialogue_history_items": 100,
    }


@pytest.fixture
def sample_workflow_data():
    """Sample workflow creation data."""
    return {
        "name": "test_workflow",
        "description": "A test workflow",
        "nodes": [
            {
                "id": "node1",
                "type": "agent",
                "position": {"x": 100, "y": 100},
                "data": {"label": "Agent Node", "config": {}},
            },
            {
                "id": "node2",
                "type": "output",
                "position": {"x": 300, "y": 100},
                "data": {"label": "Output Node", "config": {}},
            },
        ],
        "edges": [{"id": "edge1", "source": "node1", "target": "node2"}],
    }
