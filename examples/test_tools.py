#!/usr/bin/env python3
"""
Simple test script to demonstrate the mock email and note creator tools.
This can be run independently to test tool functionality.
"""

import asyncio
import os
import yaml
from dotenv import load_dotenv

from argentic.core.messager.messager import Messager
from email_tool import EmailTool
from note_creator_tool import NoteCreatorTool


async def test_tools():
    """Test the tools independently."""
    load_dotenv()

    # Load messaging config
    messaging_config_path = os.path.join(os.path.dirname(__file__), "config_messaging.yaml")
    with open(messaging_config_path, "r") as f:
        messaging_config = yaml.safe_load(f) or {}
    messaging_data = messaging_config.get("messaging", {})

    # Initialize messager
    messager = Messager(
        broker_address=messaging_data.get("broker_address", "localhost"),
        port=messaging_data.get("port", 1883),
        client_id="tool_test_client",
        keepalive=messaging_data.get("keepalive", 60),
    )

    try:
        await messager.connect()
        print("🔧 Testing Tools Independently")
        print("=" * 30)

        # Test note creator
        print("\n📝 Testing Note Creator Tool...")
        note_tool = NoteCreatorTool(messager=messager)

        note_result = await note_tool._execute(
            filename="test_report",
            content="This is a test research report about quantum computing advances.",
            folder="test_notes",
        )
        print(f"✅ {note_result}")

        # Test email tool
        print("\n📧 Testing Email Tool...")
        email_tool = EmailTool(messager=messager)

        email_result = await email_tool._execute(
            to="test@example.com",
            subject="Test Research Report",
            body="Please find the attached research report on quantum computing.\n\nBest regards,\nThe AI Secretary",
            urgent=True,
        )
        print(f"✅ {email_result}")

        print("\n🎉 All tools working correctly!")
        print("Check the 'test_notes' folder for the created file.")

    except Exception as e:
        print(f"❌ Error testing tools: {e}")
    finally:
        await messager.disconnect()


if __name__ == "__main__":
    asyncio.run(test_tools())
