from enum import Enum


class MessageSource(str, Enum):
    SYSTEM = "system"
    USER = "user"
    LLM = "llm"
    AGENT = "agent"
    CLIENT = "client"  # Added for cli_client source
    TOOL_MANAGER = "tool_manager"


class LLMRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TaskStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Add other enums as needed, for example, for message types if not already handled
