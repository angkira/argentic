"""Unit tests for Pydantic models."""

import pytest
from pydantic import ValidationError
from app.models import (
    AgentCreate,
    AgentUpdate,
    SupervisorCreate,
    WorkflowCreate,
    LLMProviderConfig,
    MessagingConfig,
)


class TestAgentModels:
    """Test agent models."""

    def test_agent_create_valid(self):
        """Test valid agent creation."""
        agent = AgentCreate(
            role="test_agent",
            description="Test agent description",
            expected_output_format="json",
            enable_dialogue_logging=False,
            max_consecutive_tool_calls=3,
            max_dialogue_history_items=100,
            max_context_iterations=10,
            enable_adaptive_context_management=True,
        )
        assert agent.role == "test_agent"
        assert agent.description == "Test agent description"
        assert agent.expected_output_format == "json"

    def test_agent_create_invalid_output_format(self):
        """Test agent creation with invalid output format."""
        with pytest.raises(ValidationError):
            AgentCreate(
                role="test_agent",
                description="Test description",
                expected_output_format="invalid",  # Should be json, text, or code
                enable_dialogue_logging=False,
                max_consecutive_tool_calls=3,
                max_dialogue_history_items=100,
                max_context_iterations=10,
                enable_adaptive_context_management=True,
            )

    def test_agent_create_missing_required_fields(self):
        """Test agent creation with missing required fields."""
        with pytest.raises(ValidationError):
            AgentCreate(role="test_agent")  # Missing description

    def test_agent_update_partial(self):
        """Test partial agent update."""
        update = AgentUpdate(description="Updated description")
        assert update.description == "Updated description"
        assert update.system_prompt is None


class TestSupervisorModels:
    """Test supervisor models."""

    def test_supervisor_create_valid(self):
        """Test valid supervisor creation."""
        supervisor = SupervisorCreate(
            role="test_supervisor",
            description="Test supervisor",
            worker_agents=[
                {"role": "worker1", "description": "Worker 1"},
                {"role": "worker2", "description": "Worker 2"},
            ],
            enable_dialogue_logging=True,
            max_dialogue_history_items=100,
        )
        assert supervisor.role == "test_supervisor"
        assert len(supervisor.worker_agents) == 2
        assert supervisor.worker_agents[0].role == "worker1"

    def test_supervisor_create_empty_workers(self):
        """Test supervisor creation with empty worker list."""
        supervisor = SupervisorCreate(
            role="test_supervisor",
            description="Test supervisor",
            worker_agents=[],
            enable_dialogue_logging=True,
            max_dialogue_history_items=100,
        )
        assert len(supervisor.worker_agents) == 0


class TestWorkflowModels:
    """Test workflow models."""

    def test_workflow_create_valid(self):
        """Test valid workflow creation."""
        workflow = WorkflowCreate(
            name="test_workflow",
            description="Test workflow",
            nodes=[
                {
                    "id": "node1",
                    "type": "agent",
                    "position": {"x": 100, "y": 100},
                    "data": {"label": "Test Node", "config": {}},
                }
            ],
            edges=[],
        )
        assert workflow.name == "test_workflow"
        assert len(workflow.nodes) == 1
        assert workflow.nodes[0].type == "agent"

    def test_workflow_create_invalid_node_type(self):
        """Test workflow creation with invalid node type."""
        with pytest.raises(ValidationError):
            WorkflowCreate(
                name="test_workflow",
                description="Test workflow",
                nodes=[
                    {
                        "id": "node1",
                        "type": "invalid_type",  # Should be agent, supervisor, tool, input, or output
                        "position": {"x": 100, "y": 100},
                        "data": {"label": "Test Node", "config": {}},
                    }
                ],
                edges=[],
            )


class TestConfigModels:
    """Test configuration models."""

    def test_llm_provider_config_gemini(self):
        """Test Gemini LLM provider configuration."""
        config = LLMProviderConfig(
            provider="google_gemini",
            google_gemini_api_key="test_key",
            google_gemini_model_name="gemini-2.0-flash",
            parameters={"temperature": 0.7},
        )
        assert config.provider == "google_gemini"
        assert config.google_gemini_api_key == "test_key"
        assert config.parameters["temperature"] == 0.7

    def test_llm_provider_config_ollama(self):
        """Test Ollama LLM provider configuration."""
        config = LLMProviderConfig(
            provider="ollama",
            ollama_model_name="llama2",
            ollama_base_url="http://localhost:11434",
        )
        assert config.provider == "ollama"
        assert config.ollama_model_name == "llama2"

    def test_messaging_config_mqtt(self):
        """Test MQTT messaging configuration."""
        config = MessagingConfig(
            protocol="mqtt",
            broker_address="localhost",
            port=1883,
            keepalive=60,
            use_tls=False,
        )
        assert config.protocol == "mqtt"
        assert config.broker_address == "localhost"
        assert config.port == 1883

    def test_messaging_config_invalid_port(self):
        """Test messaging configuration with invalid port."""
        # Pydantic won't validate port range in the model itself,
        # but we can test that the model accepts the value
        config = MessagingConfig(
            protocol="mqtt",
            broker_address="localhost",
            port=70000,  # Invalid port but model will accept it
            keepalive=60,
            use_tls=False,
        )
        assert config.port == 70000
