import asyncio
from typing import List, Optional, Any, Dict, Union, Hashable
import functools
import json

from langchain_core.messages import BaseMessage as LangchainBaseMessage, AIMessage, ToolMessage
from pydantic import BaseModel, create_model
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END


from argentic.core.agent.agent import (
    Agent,
)
from argentic.core.messager.messager import Messager
from argentic.core.tools.tool_manager import ToolManager
from argentic.core.protocol.message import (
    AgentLLMRequestMessage,
    AgentLLMResponseMessage,
    AskQuestionMessage,
    AnswerMessage,
    MinimalToolCallRequest,
)
from argentic.core.protocol.task import TaskResultMessage, TaskErrorMessage
from argentic.core.protocol.enums import MessageSource, LLMRole
from argentic.core.logger import get_logger, LogLevel, parse_log_level
from argentic.core.graph.state import AgentState
from langchain_core.messages import HumanMessage


class Supervisor(Agent):
    def __init__(
        self,
        llm: Any,
        messager: Messager,
        tool_manager: ToolManager,
        log_level: Union[str, LogLevel] = LogLevel.INFO,
        register_topic: str = "agent/tools/register",
        tool_call_topic_base: str = "agent/tools/call",
        tool_response_topic_base: str = "agent/tools/response",
        status_topic: str = "agent/status/info",
        answer_topic: str = "agent/response/answer",
        llm_response_topic: Optional[str] = None,
        tool_result_topic: Optional[str] = None,
        system_prompt: Optional[str] = None,
        role: str = "supervisor",
        graph_id: Optional[str] = None,
        agents: Optional[List[Agent]] = None,
    ):
        self.agent_tools: List[Any] = []
        self.tool_manager = tool_manager
        self._tool_id_map: Dict[str, str] = {}

        super().__init__(
            llm=llm,
            messager=messager,
            log_level=log_level,
            register_topic=register_topic,
            tool_call_topic_base=tool_call_topic_base,
            tool_response_topic_base=tool_response_topic_base,
            status_topic=status_topic,
            answer_topic=answer_topic,
            llm_response_topic=llm_response_topic,
            tool_result_topic=tool_result_topic,
            system_prompt=system_prompt,  # Keep no custom prompt for simplicity
            role=role,
            graph_id=graph_id,
            expected_output_format="text",  # Revert to text format
        )
        self.logger = get_logger("supervisor", self.log_level)
        self._agents: Dict[str, Agent] = {a.role: a for a in agents} if agents else {}
        self._graph = StateGraph(AgentState)
        self._graph.add_node("supervisor", self.invoke)
        self.runnable: Optional[Any] = None

    def add_agent(self, agent: Agent) -> None:
        self._agents[agent.role] = agent
        self._graph.add_node(agent.role, agent.invoke)
        self._graph.add_edge(agent.role, "supervisor")

    def set_available_tools(self) -> None:
        self.logger.info(f"Supervisor '{self.role}': Updating available tools.")
        langchain_tools = []
        self._tool_id_map.clear()
        for tool_id, tool_info in self.tool_manager.tools_by_id.items():
            tool_name: str = tool_info["name"]
            description: str = tool_info["description"]
            parameters_str: str = tool_info["parameters"]
            self._tool_id_map[tool_name] = tool_id

            try:
                parameters: Dict[str, Any] = json.loads(parameters_str)
            except (json.JSONDecodeError, TypeError):
                self.logger.error(
                    f"Failed to parse tool parameters for '{tool_name}'. Got: {parameters_str}"
                )
                continue

            def create_tool_callable(t_name: str, t_id: str):
                async def async_tool_func(**kwargs: Any) -> Any:
                    self.logger.info(
                        f"Supervisor delegating to tool '{t_name}' (ID: {t_id}) with args: {kwargs}"
                    )
                    result_msg = await self.tool_manager.execute_tool(t_id, kwargs)
                    if isinstance(result_msg, TaskResultMessage):
                        return result_msg.result
                    elif isinstance(result_msg, TaskErrorMessage):
                        return f"Error from tool '{t_name}': {result_msg.error}"
                    return f"Unknown result from tool '{t_name}'"

                tool_args_fields: Dict[str, Any] = {
                    p_name: (Any, None) for p_name in parameters.get("properties", {}).keys()
                }
                ToolArgsModel = create_model(f"{t_name.capitalize()}Args", **tool_args_fields)

                return Tool(
                    name=t_name,
                    description=description,
                    func=None,
                    coroutine=async_tool_func,
                    args_schema=ToolArgsModel,
                )

            langchain_tools.append(create_tool_callable(tool_name, tool_id))

        self.agent_tools = langchain_tools
        self.logger.info(f"Rebuilt with {len(self.agent_tools)} tools. Map: {self._tool_id_map}")
        self.prompt_template = self._build_prompt_template()

    async def invoke(self, state: AgentState) -> dict[str, list[LangchainBaseMessage]]:
        """
        Supervisor only routes messages to appropriate agents, it doesn't answer questions directly.
        """
        messages: List[LangchainBaseMessage] = state["messages"]
        self.logger.info(f"Agent '{self.role}' invoked with {len(messages)} messages.")

        # Check if this is the first message (from human) - route directly
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            self.logger.info("Initial human message - routing to appropriate agent.")
            return {"messages": messages}

        # For subsequent messages, we're processing agent responses
        # Just pass them through without generating new content
        self.logger.info("Processing agent response - preparing for next routing decision.")
        return {"messages": messages}

    def compile(self) -> Any:
        self.logger.info(f"Supervisor '{self.role}': Compiling graph.")
        self._graph.set_entry_point("supervisor")
        path_map: Dict[Hashable, str] = {
            agent_role: agent_role for agent_role in self._agents.keys()
        }
        path_map["__end__"] = "__end__"
        path_map["supervisor"] = "supervisor"

        self._graph.add_conditional_edges("supervisor", self._route, path_map)
        self.runnable = self._graph.compile()
        self.logger.info(f"Supervisor '{self.role}': Graph compiled successfully.")
        return self.runnable

    async def _route(self, state: AgentState) -> str:
        messages: List[LangchainBaseMessage] = state["messages"]
        last_message = messages[-1]

        # If this is the initial human message, route to the most appropriate agent
        if len(messages) == 1 and isinstance(last_message, HumanMessage):
            content = last_message.content if last_message.content else ""
            if not isinstance(content, str):
                content = str(content)

            # Use LLM to determine initial routing
            available_agents = list(self._agents.keys())
            if not available_agents:
                self.logger.info("No agents available, routing to __end__")
                return "__end__"

            try:
                self.logger.info(f"Supervisor: Starting initial routing decision...")
                routing_decision = await self._llm_route_decision(content, available_agents)
                self.logger.info(f"Initial routing decision: {routing_decision}")
                return routing_decision
            except Exception as e:
                self.logger.error(
                    f"Error in initial routing: {e}, routing to first available agent"
                )
                return available_agents[0] if available_agents else "__end__"

        # If we have agent responses, end the conversation
        # (Agents should provide complete answers, no need for supervisor to process them)
        if len(messages) > 1 and isinstance(last_message, AIMessage):
            self.logger.info("Agent has provided response - ending conversation")
            return "__end__"

        # Fallback
        self.logger.info("Unexpected state - ending conversation")
        return "__end__"

    async def _llm_route_decision(self, content: str, available_agents: List[str]) -> str:
        """Use LLM to determine which agent should handle the message."""

        # Create routing prompt
        agent_descriptions = {
            "researcher": "Handles research tasks, finding information, data analysis, web searches, and knowledge retrieval",
            "coder": "Handles programming tasks, code development, debugging, script writing, and technical implementation",
        }

        agents_info = []
        for agent in available_agents:
            desc = agent_descriptions.get(agent, f"Handles {agent} related tasks")
            agents_info.append(f"- {agent}: {desc}")

        routing_prompt = f"""You are a routing assistant. Analyze the following message and determine which agent should handle it.

Available agents:
{chr(10).join(agents_info)}

Message to route: "{content}"

Rules:
1. Choose the most appropriate agent from the available list
2. If the message is a final answer or conclusion, respond with "__end__"
3. If unsure or the message needs supervisor analysis, respond with "supervisor"
4. Respond with ONLY the agent name, nothing else

Agent to route to:"""

        # Prepare message for LLM
        llm_messages = [{"role": "user", "content": routing_prompt}]

        # Call LLM
        response = await self._call_llm(llm_messages)

        # Extract routing decision
        if hasattr(response, "content") and response.content:
            # Handle both string and list content types
            if isinstance(response.content, str):
                decision = response.content.strip().lower()
            elif isinstance(response.content, list):
                # Join list elements or take first element if it's a list
                decision = str(response.content[0] if response.content else "").strip().lower()
            else:
                decision = str(response.content).strip().lower()

            # Validate the decision
            valid_options = available_agents + ["__end__", "supervisor"]
            valid_options_lower = [opt.lower() for opt in valid_options]

            if decision in valid_options_lower:
                # Return the properly cased version
                for opt in valid_options:
                    if opt.lower() == decision:
                        return opt

            # If decision contains one of the valid options, extract it
            for opt in valid_options:
                if opt.lower() in decision:
                    return opt

        # Fallback to supervisor if we can't determine
        content_str = ""
        if hasattr(response, "content"):
            if isinstance(response.content, str):
                content_str = response.content
            else:
                content_str = str(response.content)
        self.logger.warning(
            f"Could not determine routing from LLM response: {content_str or 'No content'}"
        )
        return "supervisor"

    def _get_default_supervisor_prompt(self) -> str:
        """
        Returns the default system prompt for the supervisor.
        """
        agent_names = list(self._agents.keys()) if self._agents else []
        agent_list = ", ".join(agent_names) if agent_names else "none"

        return f"""You are a supervisor agent coordinating tasks among specialized agents: {agent_list}

Your role:
1. Analyze incoming requests and coordinate with specialized agents
2. Use available tools when needed for direct execution
3. Synthesize results from agents to provide comprehensive answers
4. Ensure tasks are completed effectively

Available specialized agents:
- researcher: Handles research, information gathering, data analysis, and knowledge retrieval
- coder: Handles programming tasks, development, debugging, and technical implementation

Be direct and efficient in your coordination."""
