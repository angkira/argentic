import os

import httpx
from dotenv import load_dotenv
from pydantic.main import BaseModel

from argentic.core.logger import LogLevel
from argentic.core.tools.tool_base import BaseTool

load_dotenv()


class GoogleSearchToolSchema(BaseModel):
    query: str


class GoogleSearchTool(BaseTool):
    def __init__(self, messager, log_level=LogLevel.INFO):
        super().__init__(
            name="google_search",
            manual="Searches Google. Argument: 'query' - search query string",
            api='{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}',
            argument_schema=GoogleSearchToolSchema,
            messager=messager,
        )
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")

        if not self.api_key or not self.cse_id:
            raise ValueError("GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables must be set")

    async def _execute(self, **kwargs) -> str:
        query = kwargs["query"]

        # Google Custom Search API endpoint
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": 5,  # Number of results to return
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()

                data = response.json()

                # Format the results
                if "items" not in data:
                    return f"No search results found for: {query}"

                results = []
                for item in data["items"]:
                    title = item.get("title", "No title")
                    link = item.get("link", "")
                    snippet = item.get("snippet", "No description")
                    results.append(f"**{title}**\n{snippet}\nURL: {link}\n")

                return "\n".join(results)

            except httpx.HTTPStatusError as e:
                return f"Google Search API error: {e.response.status_code} - {e.response.text}"
            except Exception as e:
                return f"Google Search error: {str(e)}"
