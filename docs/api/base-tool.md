# BaseTool API Reference

The `BaseTool` class provides the foundation for creating custom tools in the Argentic framework with automatic messaging, validation, and error handling.

## Class Definition

```python
from abc import ABC, abstractmethod
from typing import Any, Type
from pydantic import BaseModel
from argentic.core.messager.messager import Messager

class BaseTool(ABC):
    def __init__(
        self,
        name: str,
        manual: str,
        api: str,
        argument_schema: Type[BaseModel],
        messager: Messager,
    )
```

## Parameters

#### `name: str` (required)
Human-readable name for the tool (e.g., "Email Sender", "File Creator").

#### `manual: str` (required)
Description of what the tool does and how to use it. This becomes the LLM function description.

#### `api: str` (required)
JSON schema string defining the tool's parameters and their types.

#### `argument_schema: Type[BaseModel]` (required)
Pydantic model class for validating tool arguments.

#### `messager: Messager` (required)
Messaging system instance for MQTT communication.

## Properties

#### `id: str`
Unique tool identifier assigned during registration (read-only).

#### `task_topic: str`
MQTT topic where the tool receives task messages (set after registration).

#### `result_topic: str`
MQTT topic where the tool publishes results (set after registration).

#### `logger: Logger`
Tool-specific logger instance for debugging and monitoring.

## Abstract Methods

#### `async def _execute(self, **kwargs) -> Any`
**Must be implemented by subclasses.**

Core tool execution logic that processes validated arguments and returns results.

**Parameters:**
- `**kwargs`: Validated arguments from the Pydantic schema

**Returns:** Any serializable result (string, dict, list, etc.)

**Example:**
```python
async def _execute(self, message: str, recipient: str) -> str:
    # Send email logic here
    return f"Email sent to {recipient}: {message}"
```

## Methods

### Registration Methods

#### `async def register(self, registration_topic: str, status_topic: str, call_topic_base: str, response_topic_base: str) -> None`
Register the tool with the messaging system.

**Parameters:**
- `registration_topic`: Topic for tool registration requests
- `status_topic`: Topic for registration confirmation
- `call_topic_base`: Base topic for incoming tool calls  
- `response_topic_base`: Base topic for publishing results

**Example:**
```python
await my_tool.register(
    registration_topic="agent/tools/register",
    status_topic="agent/status/info", 
    call_topic_base="agent/tools/call",
    response_topic_base="agent/tools/response"
)
```

#### `async def unregister(self) -> None`
Unregister the tool from the messaging system.

### Utility Methods

#### `def get_definition_for_prompt(self) -> Dict[str, Any]`
Generate LLM function calling definition for the tool.

**Returns:** Dictionary with tool definition in LLM function format

## Tool Development Guide

### 1. Create Argument Schema

Define a Pydantic model for your tool's parameters:

```python
from pydantic import BaseModel, Field
from typing import Optional

class EmailArguments(BaseModel):
    recipient: str = Field(description="Email address of the recipient")
    subject: str = Field(description="Subject line of the email")  
    message: str = Field(description="Email message content")
    cc: Optional[str] = Field(None, description="Optional CC recipients")
```

### 2. Implement Tool Class

Extend `BaseTool` and implement the `_execute` method:

```python
from argentic.core.tools.tool_base import BaseTool
from argentic.core.messager.messager import Messager

class EmailTool(BaseTool):
    def __init__(self, messager: Messager):
        api_schema = EmailArguments.model_json_schema()
        super().__init__(
            name="Email Sender",
            manual="Send emails to specified recipients with subject and message content",
            api=json.dumps(api_schema),
            argument_schema=EmailArguments,
            messager=messager,
        )
    
    async def _execute(self, recipient: str, subject: str, message: str, cc: Optional[str] = None) -> str:
        # Implement email sending logic
        try:
            # Send email using your preferred service/library
            send_email(recipient, subject, message, cc)
            return f"✅ Email sent successfully to {recipient}"
        except Exception as e:
            raise Exception(f"Failed to send email: {str(e)}")
```

### 3. Register and Use Tool

```python
# Initialize messaging
messager = Messager(broker_address="localhost", port=1883)
await messager.connect()

# Create and register tool
email_tool = EmailTool(messager)
await email_tool.register(
    registration_topic="agent/tools/register",
    status_topic="agent/status/info",
    call_topic_base="agent/tools/call", 
    response_topic_base="agent/tools/response"
)
```

## Complete Example: File Creator Tool

```python
import json
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from argentic.core.tools.tool_base import BaseTool
from argentic.core.messager.messager import Messager

# 1. Define argument schema
class FileArguments(BaseModel):
    filename: str = Field(description="Name of the file to create")
    content: str = Field(description="Content to write to the file")
    directory: Optional[str] = Field("./", description="Directory path (default: current)")
    overwrite: Optional[bool] = Field(False, description="Whether to overwrite existing files")

# 2. Implement tool
class FileCreatorTool(BaseTool):
    def __init__(self, messager: Messager):
        api_schema = FileArguments.model_json_schema()
        super().__init__(
            name="File Creator",
            manual="Create text files with specified content in a given directory",
            api=json.dumps(api_schema),
            argument_schema=FileArguments,
            messager=messager,
        )
    
    async def _execute(
        self, 
        filename: str, 
        content: str, 
        directory: str = "./", 
        overwrite: bool = False
    ) -> str:
        try:
            # Create directory if it doesn't exist
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create file path
            file_path = dir_path / filename
            
            # Check if file exists and overwrite setting
            if file_path.exists() and not overwrite:
                raise Exception(f"File {filename} already exists. Set overwrite=true to replace it.")
            
            # Write content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"✅ File '{filename}' created successfully in {directory}"
            
        except Exception as e:
            raise Exception(f"Failed to create file '{filename}': {str(e)}")

# 3. Usage
async def main():
    messager = Messager(broker_address="localhost", port=1883)
    await messager.connect()
    
    # Create and register tool
    file_tool = FileCreatorTool(messager)
    await file_tool.register(
        registration_topic="agent/tools/register",
        status_topic="agent/status/info",
        call_topic_base="agent/tools/call",
        response_topic_base="agent/tools/response"
    )
    
    print("File Creator Tool registered and ready!")
```

## Advanced Features

### Error Handling

The framework provides automatic error handling:

```python
async def _execute(self, risky_parameter: str) -> str:
    if not risky_parameter:
        raise ValueError("risky_parameter cannot be empty")
    
    try:
        # Risky operation
        result = dangerous_operation(risky_parameter)
        return f"Success: {result}"
    except Exception as e:
        # This will be automatically caught and returned as TaskErrorMessage
        raise Exception(f"Operation failed: {str(e)}")
```

### Complex Return Types

Tools can return various data types:

```python
async def _execute(self, action: str) -> Dict[str, Any]:
    return {
        "status": "completed",
        "data": {"result": "success", "timestamp": "2024-01-01T12:00:00Z"},
        "metadata": {"tool_version": "1.0", "execution_time": 0.5}
    }
```

### Logging and Debugging

Use the built-in logger for debugging:

```python
async def _execute(self, data: str) -> str:
    self.logger.info(f"Processing data: {data[:50]}...")
    self.logger.debug(f"Full data: {data}")
    
    result = process_data(data)
    
    self.logger.info(f"Processing completed, result length: {len(result)}")
    return result
```

### Async Operations

Tools can perform async operations:

```python
import aiohttp
import asyncio

async def _execute(self, url: str, timeout: int = 30) -> str:
    self.logger.info(f"Fetching URL: {url}")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=timeout) as response:
                content = await response.text()
                return f"✅ Fetched {len(content)} characters from {url}"
        except asyncio.TimeoutError:
            raise Exception(f"Request to {url} timed out after {timeout}s")
        except Exception as e:
            raise Exception(f"Failed to fetch {url}: {str(e)}")
```

## Message Flow

The tool messaging flow:

```
1. Tool registers → ToolManager assigns unique ID
2. ToolManager subscribes to tool's result topic  
3. Agent calls tool → ToolManager publishes TaskMessage
4. Tool receives TaskMessage → validates arguments
5. Tool executes _execute() method
6. Tool publishes TaskResultMessage or TaskErrorMessage
7. ToolManager receives result → forwards to Agent
```

## Best Practices

### 1. Descriptive Names and Documentation

```python
# Good
name="Web Content Fetcher"
manual="Fetch and return the text content of web pages via HTTP GET requests"

# Poor  
name="Fetcher"
manual="Gets stuff"
```

### 2. Comprehensive Validation

```python
class WebFetchArguments(BaseModel):
    url: str = Field(description="Valid HTTP/HTTPS URL to fetch", pattern=r'^https?://')
    timeout: int = Field(default=30, ge=1, le=300, description="Timeout in seconds (1-300)")
    max_size: int = Field(default=1048576, ge=1024, description="Max response size in bytes")
```

### 3. Clear Error Messages

```python
async def _execute(self, url: str) -> str:
    if not url.startswith(('http://', 'https://')):
        raise ValueError("URL must start with http:// or https://")
    
    try:
        # Operation
        pass
    except ConnectionError:
        raise Exception(f"Could not connect to {url} - check your internet connection")
    except Exception as e:
        raise Exception(f"Unexpected error fetching {url}: {str(e)}")
```

### 4. Resource Management

```python
async def _execute(self, filename: str) -> str:
    file_handle = None
    try:
        file_handle = open(filename, 'r')
        content = file_handle.read()
        return f"File contains {len(content)} characters"
    except FileNotFoundError:
        raise Exception(f"File {filename} not found")
    finally:
        if file_handle:
            file_handle.close()
```

### 5. Idempotent Operations

Design tools to be safely retryable:

```python
async def _execute(self, directory: str) -> str:
    # Safe to call multiple times
    Path(directory).mkdir(parents=True, exist_ok=True)
    return f"✅ Directory {directory} ensured to exist"
```

## Testing Tools

Test tools independently:

```python
import pytest
from unittest.mock import Mock

@pytest.mark.asyncio
async def test_email_tool():
    # Mock messager
    mock_messager = Mock()
    
    # Create tool
    tool = EmailTool(mock_messager)
    
    # Test execution
    result = await tool._execute(
        recipient="test@example.com",
        subject="Test",
        message="Hello World"
    )
    
    assert "Email sent successfully" in result
```

## Performance Considerations

### 1. Timeout Handling

```python
async def _execute(self, long_task: str) -> str:
    # For long-running tasks, provide progress updates via logging
    self.logger.info("Starting long-running operation...")
    
    # Use asyncio.timeout for internal timeouts
    try:
        async with asyncio.timeout(60):  # 1 minute internal timeout
            result = await long_operation(long_task)
            return result
    except asyncio.TimeoutError:
        raise Exception("Operation timed out after 60 seconds")
```

### 2. Memory Management

```python
async def _execute(self, large_file: str) -> str:
    # Process large files in chunks
    chunk_size = 8192
    total_size = 0
    
    with open(large_file, 'rb') as f:
        while chunk := f.read(chunk_size):
            total_size += len(chunk)
            # Process chunk without loading entire file
    
    return f"Processed {total_size} bytes"
```

## See Also

- [Agent API](agent.md) - Using tools in agents
- [ToolManager API](tool-manager.md) - Tool management system  
- [Messaging Configuration](../messaging-configuration.md) - MQTT setup
- [Examples](../../examples/) - Complete tool examples 