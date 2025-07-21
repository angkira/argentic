import json
import os
import time
from typing import Optional, Any
from pydantic import BaseModel, Field

from argentic.core.tools.tool_base import BaseTool
from argentic.core.messager.messager import Messager
from argentic.core.logger import get_logger, LogLevel, parse_log_level


class NoteInput(BaseModel):
    filename: str = Field(description="Name of the file to create (without extension)")
    content: str = Field(description="Content to save in the note")
    folder: Optional[str] = Field("notes", description="Folder to save the note in")
    append: Optional[bool] = Field(
        False, description="Append to existing file instead of overwriting"
    )


class NoteCreatorTool(BaseTool):
    """Tool for creating and saving text notes to local files."""

    def __init__(
        self,
        messager: Messager,
        log_level: LogLevel = LogLevel.INFO,
    ):
        api_schema = NoteInput.model_json_schema()
        super().__init__(
            name="note_creator_tool",
            manual=(
                "Creates and saves text notes to local files. Provide 'filename' (without extension), "
                "'content' to save. Optionally specify 'folder' (defaults to 'notes') and 'append' "
                "flag to append to existing file."
            ),
            api=json.dumps(api_schema),
            argument_schema=NoteInput,
            messager=messager,
        )
        self.log_level = log_level
        self.logger = get_logger("note_creator", self.log_level)
        self.logger.info("NoteCreatorTool initialized")

    async def _execute(self, **kwargs) -> Any:
        """Execute note creation."""

        filename = kwargs.get("filename")
        content = kwargs.get("content")
        folder = kwargs.get("folder", "notes")
        append = kwargs.get("append", False)

        if not filename or not content:
            raise ValueError("Required fields: filename, content")

        # Ensure folder exists
        os.makedirs(folder, exist_ok=True)

        # Add .txt extension if not present
        if not filename.endswith(".txt"):
            filename += ".txt"

        filepath = os.path.join(folder, filename)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Prepare content with timestamp
        note_content = f"=== Note Created: {timestamp} ===\n{content}\n\n"

        mode = "a" if append else "w"

        try:
            with open(filepath, mode, encoding="utf-8") as f:
                f.write(note_content)

            action_word = "appended to" if append else "created"
            self.logger.info(f"📝 Note {action_word}: {filepath}")

            result = f"✅ Note successfully {action_word} '{filepath}'"
            if append:
                result += " (content appended)"

            return result

        except Exception as e:
            error_msg = f"Failed to save note to '{filepath}': {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
