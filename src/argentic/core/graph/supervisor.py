import asyncio
from typing import List, Optional, Any, Dict, Union
import uuid
import json
import time

from argentic.core.messager.messager import Messager
from argentic.core.protocol.message import (
    AgentTaskMessage,
    AgentTaskResultMessage,
    BaseMessage,
)
from argentic.core.protocol.enums import MessageSource
from argentic.core.logger import get_logger, LogLevel, parse_log_level
from argentic.core.llm.providers.base import ModelProvider


class Supervisor:
    """Pure messaging-based coordinator for multi-agent workflows."""

    def __init__(
        self,
        llm: ModelProvider,
        messager: Messager,
        log_level: Union[str, LogLevel] = LogLevel.INFO,
        role: str = "supervisor",
        system_prompt: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        enable_dialogue_logging: bool = False,  # Add dialogue logging
        max_task_history_items: int = 10,  # Context management
        max_dialogue_history_items: int = 50,  # Dialogue history limit
        context_cleanup_threshold: int = 100,  # When to trigger cleanup
    ):
        self.llm = llm
        self.messager = messager
        self.role = role
        self.system_prompt = f"{self._get_default_system_prompt()}\n{system_prompt}"
        self.llm_config = llm_config or {}

        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level

        self.logger = get_logger("supervisor", self.log_level)

        # Agent management
        self._agents: Dict[str, str] = {}  # role -> description mapping

        # Task tracking
        self._active_tasks: Dict[str, Dict[str, Any]] = {}  # task_id -> task_info
        self._completion_callbacks: Dict[str, Any] = {}  # task_id -> callback

        # Dialogue logging
        self.enable_dialogue_logging = enable_dialogue_logging
        self.dialogue_history: List[Dict[str, Any]] = []

        # Context management for endless cycles
        self.max_task_history_items = max_task_history_items
        self.max_dialogue_history_items = max_dialogue_history_items
        self.context_cleanup_threshold = context_cleanup_threshold
        self._total_tasks_processed = 0

        self.logger.info(f"Supervisor '{self.role}': Initialized pure messaging coordinator")

    def _cleanup_context_if_needed(self) -> None:
        """Clean up context to prevent memory and context window overflow."""
        # Clean up dialogue history
        if len(self.dialogue_history) > self.max_dialogue_history_items:
            # Keep most recent entries
            excess = len(self.dialogue_history) - self.max_dialogue_history_items
            self.dialogue_history = self.dialogue_history[excess:]
            self.logger.info(f"Supervisor: Cleaned up {excess} old dialogue entries")

        # Periodic cleanup of completed task artifacts
        if self._total_tasks_processed % self.context_cleanup_threshold == 0:
            self._deep_cleanup()

    def _deep_cleanup(self) -> None:
        """Perform deep cleanup of accumulated data."""
        # Clean up any orphaned callbacks
        orphaned_callbacks = []
        for task_id in list(self._completion_callbacks.keys()):
            if task_id not in self._active_tasks:
                orphaned_callbacks.append(task_id)

        for task_id in orphaned_callbacks:
            del self._completion_callbacks[task_id]

        if orphaned_callbacks:
            self.logger.info(f"Supervisor: Cleaned up {len(orphaned_callbacks)} orphaned callbacks")

    def _truncate_task_history(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate task history to prevent context overflow while preserving essential info."""
        if "history" in task_info and len(task_info["history"]) > self.max_task_history_items:
            # Keep first item (for context) and recent items
            history = task_info["history"]
            if len(history) <= self.max_task_history_items:
                return task_info

            # Keep first 2 and last (max_task_history_items - 2) items
            keep_first = min(2, len(history))
            keep_recent = self.max_task_history_items - keep_first

            truncated_history = history[:keep_first] + history[-keep_recent:]

            # Add truncation marker
            if len(history) > len(truncated_history):
                marker = {
                    "agent": "system",
                    "result": f"[TRUNCATED: {len(history) - len(truncated_history)} entries omitted]",
                    "success": True,
                    "error": None,
                }
                truncated_history.insert(keep_first, marker)

            task_info = task_info.copy()
            task_info["history"] = truncated_history

        return task_info

    def _log_dialogue(self, role: str, content: str, message_type: str = "routing") -> None:
        """Log a dialogue entry for the supervisor."""
        if not self.enable_dialogue_logging:
            return

        dialogue_entry = {
            "timestamp": time.time(),
            "role": role,
            "message_type": message_type,
            "content_preview": self._get_content_preview(content),
            "agent": self.role,
        }

        self.dialogue_history.append(dialogue_entry)

        # Cleanup dialogue history immediately if it exceeds the limit
        self._cleanup_context_if_needed()

        self._print_dialogue_entry(dialogue_entry)

    def _get_content_preview(self, content: str, max_length: int = 200) -> str:
        """Get a preview of content limited to max_length characters."""
        if not content:
            return "[Empty]"

        clean_content = str(content).strip()
        if len(clean_content) > max_length:
            return clean_content[:max_length] + "..."
        return clean_content

    def _print_dialogue_entry(self, entry: Dict[str, Any]) -> None:
        """Print a supervisor dialogue entry."""
        role = entry["role"]
        preview = entry["content_preview"]
        message_type = entry["message_type"]

        if message_type == "routing":
            print(f"  👨‍💼 SUPERVISOR: Routing → {preview}")
        elif message_type == "decision":
            print(f"  👨‍💼 SUPERVISOR: Decision → {preview}")
        else:
            print(f"  👨‍💼 SUPERVISOR: {preview}")

    def add_agent(self, agent) -> None:
        """Add an agent to coordination using its role and description."""
        self._agents[agent.role] = agent.description
        self.logger.info(
            f"Supervisor: Registered agent '{agent.role}' with description: {agent.description[:100]}..."
        )

    async def async_init(self) -> None:
        """Initialize supervisor messaging subscriptions."""
        # Subscribe to task results from all agents
        result_topic = "supervisor/results"
        try:
            await self.messager.subscribe(
                result_topic, self.handle_agent_result, AgentTaskResultMessage
            )
            self.logger.info(f"Supervisor: Subscribed to result topic '{result_topic}'")
        except Exception as e:
            self.logger.error(f"Supervisor: Failed to subscribe to result topic: {e}")

        # Subscribe to initial tasks
        task_topic = "supervisor/tasks"
        try:
            await self.messager.subscribe(task_topic, self.handle_initial_task, AgentTaskMessage)
            self.logger.info(f"Supervisor: Subscribed to task topic '{task_topic}'")
        except Exception as e:
            self.logger.error(f"Supervisor: Failed to subscribe to task topic: {e}")

    async def start_task(self, task: str, completion_callback=None) -> str:
        """Start a new workflow task."""
        task_id = str(uuid.uuid4())

        # Store task info
        self._active_tasks[task_id] = {
            "original_task": task,
            "status": "routing",
            "history": [],
            "current_step": 0,
        }

        if completion_callback:
            self._completion_callbacks[task_id] = completion_callback

        self.logger.info(f"Supervisor: Starting task {task_id}: {task[:100]}...")

        # Log the initial task
        self._log_dialogue("user", f"New workflow task: {task}", "routing")

        # Route initial task
        await self._route_task(task_id, task)
        return task_id

    async def handle_initial_task(self, message: BaseMessage) -> None:
        """Handle tasks published to supervisor/tasks topic."""
        try:
            if not isinstance(message, AgentTaskMessage):
                self.logger.error(f"Expected AgentTaskMessage, got {type(message)}")
                return

            task_id = await self.start_task(message.task)
            self.logger.info(f"Supervisor: Handling initial task {task_id}")

        except Exception as e:
            self.logger.error(f"Supervisor: Error handling initial task: {e}", exc_info=True)

    async def handle_agent_result(self, message: BaseMessage) -> None:
        """Handle results from agents."""
        try:
            if not isinstance(message, AgentTaskResultMessage):
                self.logger.error(f"Expected AgentTaskResultMessage, got {type(message)}")
                return

            task_id = message.task_id

            if task_id not in self._active_tasks:
                self.logger.warning(f"Supervisor: Received result for unknown task {task_id}")
                return

            task_info = self._active_tasks[task_id]

            # Update task history
            task_info["history"].append(
                {
                    "agent": message.agent_id,
                    "result": message.result,
                    "success": message.success,
                    "error": message.error,
                }
            )

            if not message.success:
                error_msg = message.error or "Unknown error"
                self.logger.error(
                    f"Supervisor: Agent {message.agent_id} failed task {task_id}: {error_msg}"
                )
                self._log_dialogue(
                    "supervisor", f"Agent {message.agent_id} failed: {error_msg}", "decision"
                )
                await self._complete_task(task_id, success=False, error=error_msg)
                return

            self.logger.info(
                f"Supervisor: Received result from {message.agent_id} for task {task_id}"
            )

            # Log the received result
            self._log_dialogue(
                "supervisor",
                f"Received result from {message.agent_id}: {message.result[:100]}...",
                "routing",
            )

            # Determine next step
            await self._continue_or_complete_task(task_id, message.result)

        except Exception as e:
            self.logger.error(f"Supervisor: Error handling agent result: {e}", exc_info=True)

    async def _route_task(self, task_id: str, task_content: str) -> None:
        """Route task to appropriate agent using LLM decision."""
        try:
            # Use LLM to determine routing
            agent_role = await self._llm_route_decision(task_content)

            if agent_role == "__end__":
                self._log_dialogue("supervisor", "Task complete - routing to end", "decision")
                await self._complete_task(task_id, success=True)
                return

            if agent_role not in self._agents:
                self.logger.error(
                    f"Supervisor: Unknown agent role '{agent_role}' for task {task_id}"
                )
                await self._complete_task(
                    task_id, success=False, error=f"Unknown agent: {agent_role}"
                )
                return

            # Log the routing decision
            self._log_dialogue("supervisor", f"Routing to agent '{agent_role}'", "decision")

            # Send task to agent
            agent_topic = f"agent/{agent_role}/tasks"
            task_msg = AgentTaskMessage(
                task=task_content,
                task_id=task_id,
                sender_id=self.role,
                context={"step": self._active_tasks[task_id]["current_step"]},
            )

            await self.messager.publish(agent_topic, task_msg)
            self.logger.info(f"Supervisor: Routed task {task_id} to agent '{agent_role}'")

            # Update task status
            self._active_tasks[task_id]["status"] = f"with_{agent_role}"
            self._active_tasks[task_id]["current_step"] += 1

        except Exception as e:
            self.logger.error(f"Supervisor: Error routing task {task_id}: {e}", exc_info=True)
            await self._complete_task(task_id, success=False, error=str(e))

    async def _continue_or_complete_task(self, task_id: str, latest_result: str) -> None:
        """Decide whether to continue routing or complete the task."""
        try:
            task_info = self._active_tasks[task_id]
            original_task = task_info["original_task"]

            # Truncate task history to prevent context overflow
            task_info = self._truncate_task_history(task_info)

            # Extract numbered steps from the original task
            task_lines = []
            for line in original_task.split("\n"):
                line = line.strip()
                if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                    task_lines.append(line)

            # Analyze completion status
            completed_agents = [step["agent"] for step in task_info["history"]]

            # Build condensed continue prompt with truncated context
            continue_prompt = f"""WORKFLOW COMPLETION ANALYSIS

ORIGINAL TASK WITH NUMBERED STEPS:
{original_task}

EXTRACTED STEPS:
{chr(10).join([f"  {line}" for line in task_lines])}

COMPLETED WORK SO FAR ({len(task_info["history"])} steps):
{chr(10).join([f"- {step['agent']}: {'[TRUNCATED]' if '[TRUNCATED]' in str(step.get('result', '')) else 'completed'}" for step in task_info["history"]])}

AVAILABLE AGENTS:
{chr(10).join([f"- {role}: {desc[:100]}..." for role, desc in self._agents.items()])}

CURRENT SITUATION:
- Total steps identified: {len(task_lines)}
- Agents that have worked: {completed_agents}
- Latest result: {latest_result[:150]}...

CRITICAL ANALYSIS:
1. What type of work is needed for the remaining steps?
2. Which available agent's capabilities best match the remaining work?
3. Have ALL numbered steps been addressed by appropriate agents?

DECISION RULES:
- Analyze each remaining step and match it to agent capabilities
- Route to the agent whose description best fits the next required work
- ONLY use '__end__' when ALL steps are truly complete

Your response (ONLY the agent role or '__end__'):"""

            self._log_dialogue(
                "supervisor",
                f"Analyzing workflow continuation for {len(task_lines)} steps",
                "decision",
            )

            decision = await self._llm_route_decision(continue_prompt)

            if decision == "__end__":
                self._log_dialogue("supervisor", "All workflow steps completed", "decision")
                await self._complete_task(task_id, success=True, final_result=latest_result)
            else:
                # Continue with next step - build a focused task for the remaining work
                # Use only recent completed work to avoid context overflow
                recent_work = (
                    task_info["history"][-3:]
                    if len(task_info["history"]) > 3
                    else task_info["history"]
                )
                completed_work = "Recent work completed:\n" + "\n".join(
                    [f"- {step['agent']}: {str(step['result'])[:100]}..." for step in recent_work]
                )

                next_task = f"""Continue the multi-step workflow:

{original_task}

{completed_work}

Latest result from {task_info['history'][-1]['agent'] if task_info['history'] else 'previous agent'}:
{latest_result}

Please complete the remaining steps from the original task list."""

                self._log_dialogue(
                    "supervisor", f"Continuing workflow to agent '{decision}'", "decision"
                )
                await self._route_task(task_id, next_task)

        except Exception as e:
            self.logger.error(f"Supervisor: Error continuing task {task_id}: {e}", exc_info=True)
            await self._complete_task(task_id, success=False, error=str(e))

    async def _complete_task(
        self, task_id: str, success: bool, final_result: str = "", error: str = ""
    ) -> None:
        """Complete a workflow task."""
        try:
            task_info = self._active_tasks.get(task_id)
            if not task_info:
                return

            self.logger.info(f"Supervisor: Completing task {task_id} - Success: {success}")

            # Increment task counter for endless cycle management
            self._total_tasks_processed += 1

            # Call completion callback if exists
            if task_id in self._completion_callbacks:
                callback = self._completion_callbacks[task_id]
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(task_id, success, final_result, error)
                    else:
                        callback(task_id, success, final_result, error)
                except Exception as e:
                    self.logger.error(f"Supervisor: Error in completion callback: {e}")

            # Cleanup
            del self._active_tasks[task_id]
            if task_id in self._completion_callbacks:
                del self._completion_callbacks[task_id]

            # Perform context cleanup if needed (for endless cycles)
            self._cleanup_context_if_needed()

        except Exception as e:
            self.logger.error(f"Supervisor: Error completing task {task_id}: {e}", exc_info=True)

    async def _llm_route_decision(self, content: str) -> str:
        """Use LLM to determine routing decision."""
        try:
            # Build agent descriptions
            agents_info = []
            for agent_role, description in self._agents.items():
                agents_info.append(f"- **{agent_role}**: {description}")

            routing_prompt = f"""You are a task router. Route the message to the most appropriate agent based on their capabilities.

AVAILABLE AGENTS:
{chr(10).join(agents_info)}

MESSAGE TO ROUTE:
{content}

ROUTING INSTRUCTIONS:
1. Analyze the message content to understand what type of work is needed
2. Match the required capabilities with the agent descriptions above
3. Route to the agent whose description best matches the task requirements
4. Only use '__end__' when ALL steps in a multi-step workflow are truly complete
5. For multi-step tasks, route to the agent needed for the CURRENT step

RESPONSE: Provide ONLY the agent role name or '__end__'"""

            # Prepare message for LLM
            llm_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": routing_prompt},
            ]

            # Call LLM
            response = await self.llm.achat(llm_messages)

            # Extract routing decision
            if hasattr(response, "content") and response.content:
                decision = str(response.content).strip().lower()

                # Validate decision
                valid_options = list(self._agents.keys()) + ["__end__"]
                valid_options_lower = [opt.lower() for opt in valid_options]

                if decision in valid_options_lower:
                    for opt in valid_options:
                        if opt.lower() == decision:
                            return opt

                # If decision contains one of the valid options, extract it
                for opt in valid_options:
                    if opt.lower() in decision:
                        return opt

            # Fallback
            self.logger.warning(
                f"Could not determine routing from LLM response: {response.content}"
            )
            return "__end__"

        except Exception as e:
            self.logger.error(f"Supervisor: Error in LLM routing decision: {e}", exc_info=True)
            return "__end__"

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for supervisor."""
        return """You are a workflow supervisor coordinating multi-step tasks among specialized agents.

CORE RESPONSIBILITIES:
1. Route tasks to appropriate agents based on their capabilities
2. Ensure multi-step workflows are completed in the correct sequence
3. Continue routing until ALL steps in a task are finished
4. Only mark tasks complete when every requirement has been fulfilled

ROUTING PRINCIPLES:
- Analyze the task requirements carefully
- Route to the agent best suited for the CURRENT step needed
- For multi-step tasks, coordinate the sequence: research → documentation → communication
- Never complete a workflow prematurely - ensure all steps are done

Your responses must be ONLY the agent role name or '__end__'. No explanations or commentary."""
