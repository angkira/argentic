# Argentic Multi-Agent Framework Examples

This directory demonstrates the **Argentic multi-agent framework** - a sophisticated system for coordinating AI agents with shared tools and dynamic routing.

## 🏗️ **System Architecture**

The framework uses a **supervisor-worker pattern** with shared tool management:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Supervisor │◄──►│  Researcher │    │  Secretary  │
│   (Router)  │    │ (Knowledge) │    │   (Tools)   │
└─────────────┘    └─────────────┘    └─────────────┘
       │                                      │
       ▼                                      ▼
┌─────────────────────────────────────────────────────┐
│              Shared Tool Manager                    │
│  ┌─────────────┐          ┌─────────────────────┐   │
│  │ Email Tool  │          │ Note Creator Tool   │   │
│  └─────────────┘          └─────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### **Key Components:**

- **🎯 Supervisor**: Routes tasks between agents using LLM-based decision making
- **🔬 Researcher**: Specialized for information gathering and analysis
- **📝 Secretary**: Handles file operations and communication using tools
- **🛠️ Tools**: Shared utilities (email, file creation) accessible by any agent
- **⚡ Tool Manager**: Coordinates tool registration and execution across agents

## 🔄 **Workflow Process**

### **Complete Task Flow:**
```
User Request → Supervisor Analysis → Agent Selection → Task Execution → Tool Usage → Results
```

**Detailed Example:**
1. **Input**: "Research quantum computing and email the report to john.doe@company.com"
2. **Supervisor**: LLM analyzes request → routes to `researcher`
3. **Researcher**: Creates comprehensive research report
4. **Supervisor**: Receives report → routes to `secretary` for delivery
5. **Secretary**: Uses `note_creator_tool` to save report + `email_tool` to send
6. **Completion**: Secretary confirms both file saved and email sent

## 🚀 **Enhanced Multi-Agent Example**

The main demonstration (`multi_agent_example.py`) showcases:

### **Agents:**
- **Supervisor**: Dynamic routing with no hardcoded agent roles
- **Researcher**: Information gathering with direct, efficient prompts  
- **Secretary**: Tool execution for document management and communication

### **Tools:**
- **EmailTool**: Simulates email sending with full logging
- **NoteCreatorTool**: Creates timestamped text files with organized output

### **Workflow:**
1. User provides research task with documentation/communication requirements
2. Supervisor intelligently routes to appropriate specialist agents
3. Researcher provides comprehensive findings in structured format
4. Secretary executes tools to save reports and send communications
5. System provides real-time feedback and confirmation of all actions

## 🛠️ **Key Technical Innovations**

### **1. Shared Tool Manager**
```python
# All agents share the same tool manager instance
researcher = Agent(llm=llm, messager=messager, tool_manager=tool_manager, ...)
secretary = Agent(llm=llm, messager=messager, tool_manager=tool_manager, ...)
```

**Benefits:**
- ✅ **Unified Tool Access**: All agents can use any registered tool
- ✅ **Consistent IDs**: No tool registration conflicts
- ✅ **Efficient Resource Usage**: Single tool instance serves multiple agents

### **2. Dynamic Supervisor Routing**
```python
# Supervisor builds routing logic from actual agent system prompts
for agent_role in available_agents:
    agent = self._agents[agent_role]
    description = agent.system_prompt[:100] + "..."
    agents_info.append(f"- {agent_role}: {description}")
```

**Benefits:**
- ✅ **No Hardcoding**: Works with any agent combination
- ✅ **Self-Documenting**: Uses actual agent capabilities for routing decisions
- ✅ **Infinitely Scalable**: Add new agents without modifying supervisor code

### **3. Robust Tool Registration**
```python
# Each tool gets unique UUID and dedicated MQTT channels
tool_id = str(uuid4())  # e.g., "ba31a02c-ff33-463c-bb02-d779db398c22"
task_topic = f"agent/tools/call/{tool_id}"
result_topic = f"agent/tools/response/{tool_id}"
```

**Benefits:**
- ✅ **Message Isolation**: Each tool has private communication channels
- ✅ **Concurrent Execution**: Multiple tools can operate simultaneously
- ✅ **Error Isolation**: Tool failures don't affect other tools

## 📋 **Running the Examples**

### **Prerequisites:**
1. **Environment Setup:**
   ```bash
   # Set your Google Gemini API key
   export GOOGLE_GEMINI_API_KEY="your_api_key_here"
   ```

2. **MQTT Broker** (choose one):
   ```bash
   # Option 1: Docker (recommended)
   docker run -it -p 1883:1883 eclipse-mosquitto:2.0
   
   # Option 2: Local installation (Ubuntu/Debian)
   sudo apt install mosquitto mosquitto-clients
   sudo systemctl start mosquitto
   ```

### **Main Multi-Agent Demo:**
```bash
cd examples
python multi_agent_example.py
```

### **Tool Testing:**
```bash
# Test tools independently
python test_tools.py

# Run tools as separate service
python secretary_tools_service.py
```

### **Single Agent Demo:**
```bash
# Basic agent usage without multi-agent coordination
python single_agent_example.py
```

## 📊 **What You'll See**

### **Initialization Phase:**
```
🚀 Enhanced Multi-Agent Example
================================
📧 EmailTool initialized
📝 NoteCreatorTool initialized
✅ Tool registrations confirmed with unique IDs
👤 Agents initialized with shared tool manager
📋 Supervisor compiled with 2 tools shared across agents
```

### **Execution Phase:**
```
📋 SUPERVISOR WORKING...
   💬 Research the current status of quantum computing...
   ----------------------------------------

📋 RESEARCHER WORKING...
   💬 # Quantum Computing Update - 2024
   **TITLE**: Current Status of Quantum Computing
   **FINDINGS**: [Comprehensive research results]
   **CONCLUSION**: [Key insights and implications]
   ----------------------------------------

📋 SUPERVISOR WORKING...
   💬 Routing research results to secretary for documentation...
   ----------------------------------------

📋 SECRETARY WORKING...
   🔧 Using note_creator_tool: ✅ Note saved to notes/quantum_computing_report.txt
   📧 Using email_tool: ✅ Email sent to john.doe@company.com
   💬 Both tasks completed successfully
   ----------------------------------------

✅ TASK COMPLETED
```

### **Output Files:**
- 📁 `notes/` folder with timestamped research reports
- 📧 Console logs showing email delivery details
- 📋 Complete workflow execution summary

## ⚙️ **Configuration**

### **LLM Settings** (`config_gemini.yaml`):
```yaml
llm:
  provider: google_gemini
  google_gemini_model_name: gemini-2.0-flash-lite
```

### **Messaging Settings** (`config_messaging.yaml`):
```yaml
messaging:
  protocol: mqtt
  broker_address: localhost
  port: 1883
  keepalive: 60
```

## 🔧 **File Structure**

```
examples/
├── multi_agent_example.py      # 🎯 Main orchestrator demonstrating full workflow
├── email_tool.py              # 📧 Email simulation with comprehensive logging
├── note_creator_tool.py       # 📝 File creation with timestamp and organization
├── secretary_tools_service.py # 🔌 Standalone tool service for distributed setup
├── test_tools.py              # 🧪 Independent tool functionality testing
├── single_agent_example.py    # 👤 Basic single-agent demonstration
├── config_gemini.yaml         # 🤖 LLM provider configuration
├── config_messaging.yaml      # 📡 MQTT messaging configuration
└── README.md                  # 📖 This comprehensive documentation
```

## 💡 **Key Differences from Basic Examples**

The enhanced system provides:

- ✅ **Real Tool Integration**: Actual file creation and email simulation
- ✅ **Realistic Workflows**: Research → Documentation → Communication pipeline
- ✅ **Efficient Communication**: Direct, task-focused agent prompts
- ✅ **Professional Output**: Clean, organized results with proper formatting
- ✅ **Robust Architecture**: Shared resources and dynamic routing
- ✅ **Visual Feedback**: Real-time progress indicators and status updates

## 🚀 **Advanced Usage**

### **Adding New Agents:**
```python
# Create specialized agent
data_analyst = Agent(
    llm=llm, 
    messager=messager, 
    tool_manager=tool_manager,  # Share tools
    role="data_analyst",
    system_prompt="Analyze data and create visualizations..."
)

# Register with supervisor (no code changes needed)
supervisor.add_agent(data_analyst)
```

### **Adding New Tools:**
```python
# Create custom tool
chart_tool = ChartCreatorTool(messager=messager)
await chart_tool.register(register_topic, status_topic, call_topic_base, response_topic_base)

# Tools automatically available to all agents
```

### **Custom Workflows:**
```python
# Supervisor automatically adapts to new agent combinations
supervisor.add_agent(researcher)
supervisor.add_agent(data_analyst) 
supervisor.add_agent(report_writer)
supervisor.add_agent(quality_checker)
# LLM handles routing between all agents dynamically
```

## 🎯 **Next Steps**

1. **Experiment** with different agent combinations
2. **Create** custom tools for your specific use cases  
3. **Scale** to larger multi-agent workflows
4. **Integrate** with external APIs and services
5. **Deploy** in distributed environments using the tools service

The Argentic framework provides a solid foundation for building sophisticated multi-agent AI systems that can scale from simple demonstrations to production-ready applications. 