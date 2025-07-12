"""Services for the Argentic framework"""

# Re-export key service classes
try:
    from argentic.services.rag_tool_service import KnowledgeBaseTool

    __all__ = ["KnowledgeBaseTool"]
except ImportError:
    __all__ = []
