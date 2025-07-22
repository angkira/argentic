import os

from pydantic.main import BaseModel
from langchain_google_community import GoogleSearchAPIWrapper
from argentic.core.tools.tool_base import BaseTool
from argentic.core.logger import LogLevel

from dotenv import load_dotenv

load_dotenv()


class GoogleSearchToolSchema(BaseModel):
    query: str


class GoogleSearchTool(BaseTool):
    def __init__(self, messager, log_level=LogLevel.INFO):
        wrapper = GoogleSearchAPIWrapper(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_cse_id=os.getenv("GOOGLE_CSE_ID"),
        )
        super().__init__(
            name="google_search",
            manual="Searches Google. Argument: 'query' - search query string",
            api='{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}',
            argument_schema=GoogleSearchToolSchema,
            messager=messager,
        )
        self.wrapper = wrapper

    async def _execute(self, **kwargs):
        return self.wrapper.run(kwargs["query"])
