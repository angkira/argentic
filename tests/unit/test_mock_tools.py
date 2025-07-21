import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, Optional

from argentic.core.protocol.tool import ToolCallRequest, RegisterToolMessage
from argentic.core.protocol.task import TaskResultMessage, TaskErrorMessage, TaskStatus
from argentic.core.protocol.enums import MessageSource


class MockTool:
    """Base class for mock tools used in testing."""

    def __init__(self, tool_id: str, tool_name: str, description: str = "Mock tool for testing"):
        self.tool_id = tool_id
        self.tool_name = tool_name
        self.description = description
        self.call_count = 0
        self.call_history = []
        self.should_fail = False
        self.failure_message = "Mock tool failure"
        self.delay_seconds = 0.0

    def set_failure_mode(self, should_fail: bool, failure_message: str = "Mock tool failure"):
        """Configure the tool to fail on next execution."""
        self.should_fail = should_fail
        self.failure_message = failure_message

    def set_delay(self, delay_seconds: float):
        """Configure a delay for tool execution."""
        self.delay_seconds = delay_seconds

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the mock tool. Override in subclasses."""
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        self.call_count += 1
        self.call_history.append(arguments.copy())

        if self.should_fail:
            raise Exception(self.failure_message)

        return {"result": f"Mock tool {self.tool_name} executed with {arguments}"}

    def reset(self):
        """Reset the tool's state."""
        self.call_count = 0
        self.call_history.clear()
        self.should_fail = False

    def get_registration_message(self) -> RegisterToolMessage:
        """Get the registration message for this tool."""
        return RegisterToolMessage(
            tool_name=self.tool_name,
            tool_manual=self.description,
            tool_api='{"type": "object", "properties": {"test_param": {"type": "string"}}}',
            source=MessageSource.AGENT,
        )


class MockSearchTool(MockTool):
    """Mock search tool for testing."""

    def __init__(self):
        super().__init__(
            tool_id="search_tool",
            tool_name="search_tool",
            description="Search for information on the internet",
        )
        self.search_results = [
            "Mock search result 1: Information about the query",
            "Mock search result 2: Additional details",
            "Mock search result 3: More comprehensive data",
        ]

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search with mock results."""
        await super().execute(arguments)

        query = arguments.get("query", "default query")

        # Simulate different results based on query
        if "error" in query.lower():
            raise Exception("Search service unavailable")
        elif "empty" in query.lower():
            return {"results": [], "count": 0}
        else:
            return {
                "results": self.search_results,
                "count": len(self.search_results),
                "query": query,
            }

    def set_search_results(self, results: list):
        """Set custom search results."""
        self.search_results = results

    def get_registration_message(self) -> RegisterToolMessage:
        """Get the registration message for search tool."""
        return RegisterToolMessage(
            tool_name=self.tool_name,
            tool_manual="Search for information using a query string",
            tool_api='{"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}',
            source=MessageSource.AGENT,
        )


class MockCalculatorTool(MockTool):
    """Mock calculator tool for testing."""

    def __init__(self):
        super().__init__(
            tool_id="calculator_tool",
            tool_name="calculator_tool",
            description="Perform mathematical calculations",
        )

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute calculation with mock results."""
        await super().execute(arguments)

        expression = arguments.get("expression", "1+1")

        # Simple mock calculations
        try:
            if expression == "1+1":
                result = 2
            elif expression == "10*5":
                result = 50
            elif expression == "100/4":
                result = 25
            elif "error" in expression.lower():
                raise ValueError("Invalid expression")
            else:
                # Default result for any other expression
                result = 42

            return {"expression": expression, "result": result, "type": "number"}
        except Exception as e:
            raise Exception(f"Calculation error: {str(e)}")

    def get_registration_message(self) -> RegisterToolMessage:
        """Get the registration message for calculator tool."""
        return RegisterToolMessage(
            tool_name=self.tool_name,
            tool_manual="Perform mathematical calculations",
            tool_api='{"type": "object", "properties": {"expression": {"type": "string", "description": "Mathematical expression to evaluate"}}, "required": ["expression"]}',
            source=MessageSource.AGENT,
        )


class MockCodeExecutorTool(MockTool):
    """Mock code executor tool for testing."""

    def __init__(self):
        super().__init__(
            tool_id="code_executor",
            tool_name="code_executor",
            description="Execute code in various programming languages",
        )

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code with mock results."""
        await super().execute(arguments)

        language = arguments.get("language", "python")
        code = arguments.get("code", "print('hello')")

        # Mock execution results
        if "error" in code.lower():
            raise Exception("Code execution failed")
        elif language == "python":
            if "print('hello')" in code:
                return {"language": language, "code": code, "output": "hello\n", "exit_code": 0}
            else:
                return {
                    "language": language,
                    "code": code,
                    "output": "Mock Python execution completed\n",
                    "exit_code": 0,
                }
        else:
            return {
                "language": language,
                "code": code,
                "output": f"Mock {language} execution completed\n",
                "exit_code": 0,
            }

    def get_registration_message(self) -> RegisterToolMessage:
        """Get the registration message for code executor tool."""
        return RegisterToolMessage(
            tool_name=self.tool_name,
            tool_manual="Execute code in various programming languages",
            tool_api='{"type": "object", "properties": {"language": {"type": "string", "description": "Programming language"}, "code": {"type": "string", "description": "Code to execute"}}, "required": ["language", "code"]}',
            source=MessageSource.AGENT,
        )


class MockFileSystemTool(MockTool):
    """Mock file system tool for testing."""

    def __init__(self):
        super().__init__(
            tool_id="filesystem_tool",
            tool_name="filesystem_tool",
            description="Interact with the file system",
        )
        self.mock_files = {
            "/test/file.txt": "This is test file content",
            "/test/data.json": '{"key": "value", "number": 42}',
            "/test/README.md": "# Test Project\nThis is a test project.",
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file system operation with mock results."""
        await super().execute(arguments)

        operation = arguments.get("operation", "read")
        path = arguments.get("path", "/test/file.txt")

        if operation == "read":
            if path in self.mock_files:
                return {
                    "operation": operation,
                    "path": path,
                    "content": self.mock_files[path],
                    "success": True,
                }
            else:
                raise Exception(f"File not found: {path}")

        elif operation == "write":
            content = arguments.get("content", "")
            self.mock_files[path] = content
            return {
                "operation": operation,
                "path": path,
                "bytes_written": len(content),
                "success": True,
            }

        elif operation == "list":
            directory = path
            files = [f for f in self.mock_files.keys() if f.startswith(directory)]
            return {"operation": operation, "path": directory, "files": files, "count": len(files)}
        else:
            raise Exception(f"Unknown operation: {operation}")

    def get_registration_message(self) -> RegisterToolMessage:
        """Get the registration message for file system tool."""
        return RegisterToolMessage(
            tool_name=self.tool_name,
            tool_manual="Interact with the file system - read, write, and list files",
            tool_api='{"type": "object", "properties": {"operation": {"type": "string", "enum": ["read", "write", "list"]}, "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["operation", "path"]}',
            source=MessageSource.AGENT,
        )


class MockToolManager:
    """Mock tool manager for testing."""

    def __init__(self):
        self.tools = {}
        self.tool_results = {}
        self.call_count = 0

    def register_tool(self, tool: MockTool):
        """Register a mock tool."""
        self.tools[tool.tool_id] = tool

    def register_tools(self, tools: list[MockTool]):
        """Register multiple mock tools."""
        for tool in tools:
            self.register_tool(tool)

    async def execute_tool(self, tool_id: str, arguments: Dict[str, Any]) -> TaskResultMessage:
        """Execute a mock tool and return the result."""
        self.call_count += 1

        if tool_id not in self.tools:
            return TaskErrorMessage(
                tool_id=tool_id,
                tool_name=tool_id,
                task_id=f"task_{self.call_count}",
                error=f"Tool not found: {tool_id}",
                source=MessageSource.AGENT,
            )

        tool = self.tools[tool_id]

        try:
            result = await tool.execute(arguments)
            return TaskResultMessage(
                tool_id=tool_id,
                tool_name=tool.tool_name,
                task_id=f"task_{self.call_count}",
                status=TaskStatus.SUCCESS,
                result=result,
                source=MessageSource.AGENT,
            )
        except Exception as e:
            return TaskErrorMessage(
                tool_id=tool_id,
                tool_name=tool.tool_name,
                task_id=f"task_{self.call_count}",
                error=str(e),
                source=MessageSource.AGENT,
            )

    async def get_tool_results(self, tool_requests: list[ToolCallRequest]) -> tuple[list, bool]:
        """Execute multiple tools and return results."""
        results = []
        has_errors = False

        for request in tool_requests:
            result = await self.execute_tool(request.tool_id, request.arguments)
            results.append(result)
            if isinstance(result, TaskErrorMessage):
                has_errors = True

        return results, has_errors

    def get_tool(self, tool_id: str) -> Optional[MockTool]:
        """Get a registered tool by ID."""
        return self.tools.get(tool_id)

    def reset_all_tools(self):
        """Reset all registered tools."""
        for tool in self.tools.values():
            tool.reset()
        self.call_count = 0


class TestMockTools:
    """Test the mock tools themselves."""

    @pytest.mark.asyncio
    async def test_mock_search_tool(self):
        """Test the mock search tool."""
        tool = MockSearchTool()

        # Test normal search
        result = await tool.execute({"query": "test query"})
        assert result["count"] == 3
        assert "test query" in result["query"]
        assert tool.call_count == 1

        # Test empty results
        result = await tool.execute({"query": "empty search"})
        assert result["count"] == 0
        assert result["results"] == []

        # Test error case
        with pytest.raises(Exception):
            await tool.execute({"query": "error search"})

    @pytest.mark.asyncio
    async def test_mock_calculator_tool(self):
        """Test the mock calculator tool."""
        tool = MockCalculatorTool()

        # Test basic calculations
        result = await tool.execute({"expression": "1+1"})
        assert result["result"] == 2

        result = await tool.execute({"expression": "10*5"})
        assert result["result"] == 50

        # Test error case
        with pytest.raises(Exception):
            await tool.execute({"expression": "error expression"})

    @pytest.mark.asyncio
    async def test_mock_code_executor_tool(self):
        """Test the mock code executor tool."""
        tool = MockCodeExecutorTool()

        # Test Python execution
        result = await tool.execute({"language": "python", "code": "print('hello')"})
        assert result["output"] == "hello\n"
        assert result["exit_code"] == 0

        # Test other language
        result = await tool.execute({"language": "javascript", "code": "console.log('hello')"})
        assert "javascript" in result["output"]

    @pytest.mark.asyncio
    async def test_mock_filesystem_tool(self):
        """Test the mock file system tool."""
        tool = MockFileSystemTool()

        # Test read operation
        result = await tool.execute({"operation": "read", "path": "/test/file.txt"})
        assert result["success"] is True
        assert "test file content" in result["content"]

        # Test write operation
        result = await tool.execute(
            {"operation": "write", "path": "/test/new_file.txt", "content": "new content"}
        )
        assert result["success"] is True

        # Test list operation
        result = await tool.execute({"operation": "list", "path": "/test/"})
        assert result["count"] > 0
        assert len(result["files"]) > 0

    @pytest.mark.asyncio
    async def test_mock_tool_failure_mode(self):
        """Test mock tool failure mode."""
        tool = MockSearchTool()

        # Set failure mode
        tool.set_failure_mode(True, "Simulated failure")

        with pytest.raises(Exception, match="Simulated failure"):
            await tool.execute({"query": "test"})

    @pytest.mark.asyncio
    async def test_mock_tool_delay(self):
        """Test mock tool delay functionality."""
        tool = MockSearchTool()
        tool.set_delay(0.1)  # 100ms delay

        import time

        start_time = time.time()
        await tool.execute({"query": "test"})
        end_time = time.time()

        assert (end_time - start_time) >= 0.1

    @pytest.mark.asyncio
    async def test_mock_tool_manager(self):
        """Test the mock tool manager."""
        manager = MockToolManager()
        search_tool = MockSearchTool()
        calc_tool = MockCalculatorTool()

        manager.register_tools([search_tool, calc_tool])

        # Test tool execution
        result = await manager.execute_tool("search_tool", {"query": "test"})
        assert isinstance(result, TaskResultMessage)
        assert result.status == TaskStatus.SUCCESS

        # Test unknown tool
        result = await manager.execute_tool("unknown_tool", {})
        assert isinstance(result, TaskErrorMessage)

        # Test multiple tool execution
        tool_requests = [
            ToolCallRequest(tool_id="search_tool", arguments={"query": "test1"}),
            ToolCallRequest(tool_id="calculator_tool", arguments={"expression": "1+1"}),
        ]

        results, has_errors = await manager.get_tool_results(tool_requests)
        assert len(results) == 2
        assert not has_errors
        assert all(isinstance(r, TaskResultMessage) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
