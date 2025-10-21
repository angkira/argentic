from typing import List, Optional, TypedDict

from argentic.core.protocol.chat_message import ChatMessage


class AgentState(TypedDict):
    messages: List[ChatMessage]
    next: Optional[str]
