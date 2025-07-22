import json
import time
from typing import Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

from argentic.core.tools.tool_base import BaseTool
from argentic.core.messager.messager import Messager
from argentic.core.logger import get_logger, LogLevel, parse_log_level


class EmailInput(BaseModel):
    to: str = Field(description="Email recipient address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    cc: Optional[str] = Field(None, description="CC recipient address")
    urgent: Optional[bool] = Field(False, description="Mark email as urgent")


class EmailTool(BaseTool):
    """Email tool that simulates sending emails by logging them."""

    def __init__(
        self,
        messager: Messager,
        log_level: LogLevel = LogLevel.INFO,
    ):
        api_schema = EmailInput.model_json_schema()
        super().__init__(
            name="email_tool",
            manual=(
                "Sends emails. Provide 'to' (recipient), 'subject', and 'body'. "
                "Optionally include 'cc' for carbon copy and 'urgent' flag."
            ),
            api=json.dumps(api_schema),
            argument_schema=EmailInput,
            messager=messager,
        )
        self.log_level = log_level
        self.logger = get_logger("email_tool", self.log_level)
        self.logger.info("EmailTool initialized")

    async def _execute(self, **kwargs) -> Any:
        """Execute mock email sending."""

        to = kwargs.get("to")
        subject = kwargs.get("subject")
        body = kwargs.get("body")
        cc = kwargs.get("cc")
        urgent = kwargs.get("urgent", False)

        if not to or not subject or not body:
            raise ValueError("Required fields: to, subject, body")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Create email log entry
        email_log = f"""
=== EMAIL SENT ===
Timestamp: {timestamp}
To: {to}
{"CC: " + cc if cc else ""}
Subject: {subject}
{"[URGENT] " if urgent else ""}
Body:
{body}
=================
"""

        # Log the email
        self.logger.info(f"📧 EMAIL SENT to {to}")
        self.logger.info(f"Subject: {subject}")
        if cc:
            self.logger.info(f"CC: {cc}")
        if urgent:
            self.logger.info("⚠️ MARKED AS URGENT")

        print(email_log)  # Also print to console for visibility

        result = f"✅ Email successfully sent to {to} with subject '{subject}'"
        if cc:
            result += f" (CC: {cc})"
        if urgent:
            result += " [URGENT]"

        return result
