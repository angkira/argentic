"""Integration tests for API endpoints."""

import pytest


class TestAgentsAPI:
    """Test agents API endpoints."""

    def test_create_agent(self, client, sample_agent_data):
        """Test POST /api/agents."""
        response = client.post("/api/agents", json=sample_agent_data)
        assert response.status_code == 201
        data = response.json()
        assert data["role"] == sample_agent_data["role"]
        assert data["description"] == sample_agent_data["description"]
        assert "id" in data
        assert data["status"] == "stopped"

    def test_list_agents(self, client, sample_agent_data):
        """Test GET /api/agents."""
        # Create an agent first
        client.post("/api/agents", json=sample_agent_data)

        response = client.get("/api/agents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_get_agent(self, client, sample_agent_data):
        """Test GET /api/agents/{id}."""
        # Create an agent
        create_response = client.post("/api/agents", json=sample_agent_data)
        agent_id = create_response.json()["id"]

        # Get the agent
        response = client.get(f"/api/agents/{agent_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == agent_id
        assert data["role"] == sample_agent_data["role"]

    def test_get_nonexistent_agent(self, client):
        """Test GET /api/agents/{id} with non-existent ID."""
        response = client.get("/api/agents/nonexistent_id")
        assert response.status_code == 404

    def test_update_agent(self, client, sample_agent_data):
        """Test PATCH /api/agents/{id}."""
        # Create an agent
        create_response = client.post("/api/agents", json=sample_agent_data)
        agent_id = create_response.json()["id"]

        # Update the agent
        update_data = {"description": "Updated description"}
        response = client.patch(f"/api/agents/{agent_id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated description"
        assert data["role"] == sample_agent_data["role"]

    def test_delete_agent(self, client, sample_agent_data):
        """Test DELETE /api/agents/{id}."""
        # Create an agent
        create_response = client.post("/api/agents", json=sample_agent_data)
        agent_id = create_response.json()["id"]

        # Delete the agent
        response = client.delete(f"/api/agents/{agent_id}")
        assert response.status_code == 204

        # Verify it's deleted
        get_response = client.get(f"/api/agents/{agent_id}")
        assert get_response.status_code == 404

    def test_start_agent(self, client, sample_agent_data):
        """Test POST /api/agents/{id}/start."""
        # Create an agent
        create_response = client.post("/api/agents", json=sample_agent_data)
        agent_id = create_response.json()["id"]

        # Start the agent
        response = client.post(f"/api/agents/{agent_id}/start")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    def test_stop_agent(self, client, sample_agent_data):
        """Test POST /api/agents/{id}/stop."""
        # Create and start an agent
        create_response = client.post("/api/agents", json=sample_agent_data)
        agent_id = create_response.json()["id"]
        client.post(f"/api/agents/{agent_id}/start")

        # Stop the agent
        response = client.post(f"/api/agents/{agent_id}/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"


class TestSupervisorsAPI:
    """Test supervisors API endpoints."""

    def test_create_supervisor(self, client, sample_supervisor_data):
        """Test POST /api/supervisors."""
        response = client.post("/api/supervisors", json=sample_supervisor_data)
        assert response.status_code == 201
        data = response.json()
        assert data["role"] == sample_supervisor_data["role"]
        assert len(data["worker_agents"]) == 2

    def test_list_supervisors(self, client, sample_supervisor_data):
        """Test GET /api/supervisors."""
        client.post("/api/supervisors", json=sample_supervisor_data)

        response = client.get("/api/supervisors")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_get_supervisor(self, client, sample_supervisor_data):
        """Test GET /api/supervisors/{id}."""
        create_response = client.post("/api/supervisors", json=sample_supervisor_data)
        supervisor_id = create_response.json()["id"]

        response = client.get(f"/api/supervisors/{supervisor_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == supervisor_id


class TestWorkflowsAPI:
    """Test workflows API endpoints."""

    def test_create_workflow(self, client, sample_workflow_data):
        """Test POST /api/workflows."""
        response = client.post("/api/workflows", json=sample_workflow_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_workflow_data["name"]
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

    def test_list_workflows(self, client, sample_workflow_data):
        """Test GET /api/workflows."""
        client.post("/api/workflows", json=sample_workflow_data)

        response = client.get("/api/workflows")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_update_workflow(self, client, sample_workflow_data):
        """Test PATCH /api/workflows/{id}."""
        create_response = client.post("/api/workflows", json=sample_workflow_data)
        workflow_id = create_response.json()["id"]

        update_data = {"name": "Updated Workflow Name"}
        response = client.patch(f"/api/workflows/{workflow_id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Workflow Name"


class TestConfigAPI:
    """Test configuration API endpoints."""

    def test_list_llm_providers(self, client):
        """Test GET /api/config/llm-providers."""
        response = client.get("/api/config/llm-providers")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert any(p["name"] == "google_gemini" for p in data)
        assert any(p["name"] == "ollama" for p in data)

    def test_list_messaging_protocols(self, client):
        """Test GET /api/config/messaging-protocols."""
        response = client.get("/api/config/messaging-protocols")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert any(p["name"] == "mqtt" for p in data)

    def test_validate_llm_config_valid(self, client):
        """Test POST /api/config/llm-providers/validate with valid config."""
        config_data = {
            "provider": "google_gemini",
            "google_gemini_api_key": "test_key",
            "google_gemini_model_name": "gemini-2.0-flash",
        }
        response = client.post("/api/config/llm-providers/validate", json=config_data)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["provider"] == "google_gemini"

    def test_validate_llm_config_missing_fields(self, client):
        """Test POST /api/config/llm-providers/validate with missing fields."""
        config_data = {
            "provider": "google_gemini",
            # Missing required fields
        }
        response = client.post("/api/config/llm-providers/validate", json=config_data)
        assert response.status_code == 400

    def test_validate_messaging_config_valid(self, client):
        """Test POST /api/config/messaging/validate with valid config."""
        config_data = {
            "protocol": "mqtt",
            "broker_address": "localhost",
            "port": 1883,
            "keepalive": 60,
            "use_tls": False,
        }
        response = client.post("/api/config/messaging/validate", json=config_data)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["protocol"] == "mqtt"

    def test_validate_messaging_config_invalid_port(self, client):
        """Test POST /api/config/messaging/validate with invalid port."""
        config_data = {
            "protocol": "mqtt",
            "broker_address": "localhost",
            "port": 70000,  # Invalid port
            "keepalive": 60,
            "use_tls": False,
        }
        response = client.post("/api/config/messaging/validate", json=config_data)
        assert response.status_code == 400


class TestHealthEndpoints:
    """Test health and root endpoints."""

    def test_root_endpoint(self, client):
        """Test GET /."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["status"] == "running"

    def test_health_endpoint(self, client):
        """Test GET /health."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
