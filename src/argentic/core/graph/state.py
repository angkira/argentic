from typing import List, TypedDict, Optional
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: Optional[str]
