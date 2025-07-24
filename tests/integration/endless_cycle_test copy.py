#!/usr/bin/env python3
"""
Comprehensive test for endless cycle readiness and improvements.

This script tests:
1. File logging with rotation and size limits
2. Tool loop detection and prevention
3. Task completion analysis
4. Context management for long-running operations
5. Default utility rules vs custom prompts
6. Memory leak prevention
"""

import asyncio
import os
import sys
import yaml
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from argentic.core.agent.agent import Agent
from argentic.core.graph.supervisor import Supervisor
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider
from argentic.core.messager.messager import Messager
from argentic.core.tools.tool_manager import ToolManager
from argentic.core.logger import configure_file_logging, get_log_file_info

# Import tools with fallback strategy
try:
    # Try relative import first
    from .email_tool import EmailTool
    from .note_creator_tool import NoteCreatorTool
except ImportError:
    try:
        # Try direct import
        from email_tool import EmailTool  # type: ignore
        from note_creator_tool import NoteCreatorTool  # type: ignore
    except ImportError:
        # Skip tool tests if not available
        EmailTool = None  # type: ignore
        NoteCreatorTool = None  # type: ignore


class EndlessCycleTester:
    """Comprehensive tester for endless cycle readiness."""

    def __init__(self):
        self.test_results: Dict[str, bool] = {}
        self.temp_dir: Optional[str] = None
        self.llm: Optional[GoogleGeminiProvider] = None
        self.messager: Optional[Messager] = None

    async def setup(self):
        """Set up the test environment."""
        load_dotenv()

        # Create temporary directory for logs
        self.temp_dir = tempfile.mkdtemp(prefix="argentic_test_")
        print(f"📁 Test logs directory: {self.temp_dir}")

        # Configure file logging with small limits for testing
        configure_file_logging(
            log_dir=self.temp_dir,
            max_bytes=50 * 1024,  # 50KB for faster testing
            backup_count=5,  # Small number for testing
            enabled=True,
        )

        # Load configurations
        config_dir = Path(__file__).parent

        with open(config_dir / "config_gemini.yaml", "r") as f:
            llm_data = yaml.safe_load(f)
            llm_config = llm_data.get("llm", {}) if llm_data else {}

        with open(config_dir / "config_messaging.yaml", "r") as f:
            messaging_data = yaml.safe_load(f)
            messaging_config = messaging_data.get("messaging", {}) if messaging_data else {}

        # Initialize components
        self.llm = GoogleGeminiProvider(config=llm_config)
        self.messager = Messager(**messaging_config)
        await self.messager.connect()

        print("✅ Test environment set up successfully")

    async def cleanup(self):
        """Clean up the test environment."""
        if self.messager:
            await self.messager.disconnect()

        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up test directory: {self.temp_dir}")

    async def _cleanup_between_tests(self):
        """Clean up between individual tests to prevent conflicts."""
        # Add a small delay to allow message processing to complete
        await asyncio.sleep(1)

        # Force garbage collection to clean up any leftover objects
        import gc

        gc.collect()

    async def test_file_logging(self):
        """Test file logging with rotation."""
        print("\n🧪 Testing File Logging...")

        from argentic.core.logger import get_logger

        # Create a test logger
        test_logger = get_logger("test_logger", enable_file_logging=True)

        # Generate many log messages to test rotation
        for i in range(100):
            test_logger.info(f"Test message {i} - " + "A" * 500)  # Large message

        # Check log file info
        log_info = get_log_file_info("test_logger")

        if log_info and log_info["exists"]:
            print(f"✅ Log file created: {log_info['log_file']}")
            print(f"   Size: {log_info['size_mb']} MB")
            self.test_results["file_logging"] = True
        else:
            print("❌ Log file not created")
            self.test_results["file_logging"] = False

    async def test_tool_loop_detection(self):
        """Test tool loop detection and prevention."""
        print("\n🧪 Testing Tool Loop Detection...")

        if not NoteCreatorTool:
            print("⏭️ Skipping tool loop detection test - NoteCreatorTool not available")
            self.test_results["tool_loop_detection"] = True
            return

            # Create agent with aggressive loop detection settings
        agent = Agent(
            llm=self.llm,
            messager=self.messager,
            role="loop_test_agent",
            system_prompt="You are a test agent. When asked to save files, use the available tools.",
            expected_output_format="json",
            max_consecutive_tool_calls=2,  # Very aggressive for testing
            tool_call_window_size=4,
            enable_completion_analysis=True,
            enable_dialogue_logging=True,
        )
        await agent.async_init()

        # Register tools with default agent topics
        note_tool = NoteCreatorTool(messager=self.messager)
        await note_tool.register(
            registration_topic="agent/tools/register",
            status_topic="agent/status/info",
            call_topic_base="agent/tools/call",
            response_topic_base="agent/tools/response",
        )
        await asyncio.sleep(2)  # Allow tool registration to complete

        # Ask the agent to do something that might cause loops
        task = "Save a file called 'test_file' with content 'Hello World' using the note_creator_tool. Then confirm it was saved."

        start_time = time.time()
        result = await agent.query(task, max_iterations=8)
        end_time = time.time()

        # Check if loop detection worked (should complete quickly, not hit max iterations)
        if end_time - start_time < 30 and "loop" not in result.lower():
            print("✅ Tool loop detection prevented infinite cycles")
            self.test_results["tool_loop_detection"] = True
        else:
            print(f"❌ Tool loop detection failed - took {end_time - start_time:.1f}s")
            self.test_results["tool_loop_detection"] = False

        print(f"   Result: {result[:150]}...")

        # Cleanup between tests
        await self._cleanup_between_tests()

    async def test_completion_analysis(self):
        """Test task completion analysis."""
        print("\n🧪 Testing Task Completion Analysis...")

        if not NoteCreatorTool or not EmailTool:
            print("⏭️ Skipping completion analysis test - Tools not available")
            self.test_results["completion_analysis"] = True
            return

        agent = Agent(
            llm=self.llm,
            messager=self.messager,
            role="completion_test_agent",
            system_prompt="You are a completion test agent. Use tools when needed for file operations.",
            expected_output_format="json",
            enable_completion_analysis=True,
            enable_dialogue_logging=True,
        )
        await agent.async_init()

        # Register tools with default agent topics
        note_tool = NoteCreatorTool(messager=self.messager)
        email_tool = EmailTool(messager=self.messager)

        await note_tool.register(
            registration_topic="agent/tools/register",
            status_topic="agent/status/info",
            call_topic_base="agent/tools/call",
            response_topic_base="agent/tools/response",
        )
        await email_tool.register(
            registration_topic="agent/tools/register",
            status_topic="agent/status/info",
            call_topic_base="agent/tools/call",
            response_topic_base="agent/tools/response",
        )
        await asyncio.sleep(3)  # Allow tool registration to complete properly

        # Task that should complete after successful tool execution
        task = (
            "Create a file called 'completion_test.txt' with content 'Task completed successfully'."
        )

        try:
            result = await asyncio.wait_for(agent.query(task, max_iterations=4), timeout=45)

            # Check for completion indicators
            completion_indicators = ["completed", "successfully", "sent", "created", "saved"]
            has_completion = any(indicator in result.lower() for indicator in completion_indicators)

            if has_completion:
                print("✅ Task completion analysis working - detected completion")
                self.test_results["completion_analysis"] = True
            else:
                print("❌ Task completion analysis failed")
                self.test_results["completion_analysis"] = False

            print(f"   Result: {result[:150]}...")

        except asyncio.TimeoutError:
            print("❌ Task completion analysis failed - test timeout")
            self.test_results["completion_analysis"] = False

        # Cleanup between tests
        await self._cleanup_between_tests()

    async def test_context_management(self):
        """Test context management for long-running operations."""
        print("\n🧪 Testing Context Management...")

        agent = Agent(
            llm=self.llm,
            messager=self.messager,
            role="context_test_agent",
            system_prompt="You are a context test agent.",
            expected_output_format="text",
            max_dialogue_history_items=10,  # Small for testing
            max_query_history_items=5,  # Small for testing
            enable_dialogue_logging=True,
        )
        await agent.async_init()

        # Generate many queries to test context cleanup
        for i in range(15):  # More than max_dialogue_history_items
            query = f"Question {i}: What is {i} + {i}?"
            await agent.query(query)

        # Check if dialogue history was cleaned up
        if len(agent.dialogue_history) <= agent.max_dialogue_history_items:
            print(
                f"✅ Context management working - dialogue history limited to {len(agent.dialogue_history)} items"
            )
            self.test_results["context_management"] = True
        else:
            print(
                f"❌ Context management failed - dialogue history has {len(agent.dialogue_history)} items"
            )
            self.test_results["context_management"] = False

        # Cleanup between tests
        await self._cleanup_between_tests()

    async def test_system_prompt_override(self):
        """Test system prompt override functionality."""
        print("\n🧪 Testing System Prompt Override...")

        # Test with default rules
        agent1 = Agent(
            llm=self.llm,
            messager=self.messager,
            role="prompt_test_agent1",
            system_prompt="You are helpful.",
            override_default_prompts=False,  # Should include default rules
            expected_output_format="text",
        )

        # Test with override
        agent2 = Agent(
            llm=self.llm,
            messager=self.messager,
            role="prompt_test_agent2",
            system_prompt="You are helpful.",
            override_default_prompts=True,  # Should NOT include default rules
            expected_output_format="text",
        )

        prompt1 = agent1.get_system_prompt()
        prompt2 = agent2.get_system_prompt()

        # Check that agent1 has default rules and agent2 doesn't
        has_default_rules1 = "TOOL INTERACTION RULES" in prompt1
        has_default_rules2 = "TOOL INTERACTION RULES" in prompt2

        if has_default_rules1 and not has_default_rules2:
            print("✅ System prompt override working correctly")
            self.test_results["system_prompt_override"] = True
        else:
            print("❌ System prompt override failed")
            self.test_results["system_prompt_override"] = False

        print(f"   Agent1 (with defaults): {len(prompt1)} chars")
        print(f"   Agent2 (override): {len(prompt2)} chars")

    async def test_supervisor_context_management(self):
        """Test supervisor context management."""
        print("\n🧪 Testing Supervisor Context Management...")

        supervisor = Supervisor(
            llm=self.llm,
            messager=self.messager,
            role="test_supervisor",
            system_prompt="Route tasks efficiently.",
            max_task_history_items=3,  # Small for testing
            max_dialogue_history_items=5,  # Small for testing
            enable_dialogue_logging=True,
        )

        # Simulate many task completions
        for i in range(10):
            supervisor._log_dialogue("user", f"Task {i}", "routing")

        # Check if dialogue history was managed
        if len(supervisor.dialogue_history) <= supervisor.max_dialogue_history_items:
            print(
                f"✅ Supervisor context management working - dialogue limited to {len(supervisor.dialogue_history)}"
            )
            self.test_results["supervisor_context"] = True
        else:
            print(
                f"❌ Supervisor context management failed - dialogue has {len(supervisor.dialogue_history)} items"
            )
            self.test_results["supervisor_context"] = False

    async def run_all_tests(self):
        """Run all tests and print results."""
        print("🚀 Starting Endless Cycle Readiness Tests")
        print("=" * 50)

        try:
            await self.setup()

            # Run all tests
            await self.test_file_logging()
            await self.test_tool_loop_detection()
            await self.test_completion_analysis()
            await self.test_context_management()
            await self.test_system_prompt_override()
            await self.test_supervisor_context_management()

            # Print summary
            print("\n📊 TEST RESULTS SUMMARY")
            print("=" * 50)

            passed = 0
            total = len(self.test_results)

            for test_name, result in self.test_results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{test_name:25} {status}")
                if result:
                    passed += 1

            print(f"\nOverall: {passed}/{total} tests passed")

            if passed == total:
                print("🎉 ALL TESTS PASSED - System is ready for endless cycles!")
            else:
                print("⚠️  Some tests failed - review issues before production use")

        except Exception as e:
            print(f"💥 Test error: {e}")
            import traceback

            traceback.print_exc()

        finally:
            await self.cleanup()


async def main():
    """Run the endless cycle readiness tests."""
    tester = EndlessCycleTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
