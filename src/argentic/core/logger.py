import logging
import sys
from enum import Enum
from typing import Dict, Optional, Union


class LogLevel(Enum):
    """Enum for log levels with corresponding logging module levels"""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""

    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Wrap text with color code and reset"""
        color_code = getattr(cls, color.upper(), cls.RESET)
        return f"{color_code}{text}{cls.RESET}"


# Define color scheme for different log levels
LEVEL_COLORS = {
    LogLevel.DEBUG.value: Colors.CYAN,
    LogLevel.INFO.value: Colors.GREEN,
    LogLevel.WARNING.value: Colors.YELLOW,
    LogLevel.ERROR.value: Colors.RED,
    LogLevel.CRITICAL.value: Colors.BG_RED + Colors.WHITE + Colors.BOLD,
}

# Cache for loggers to avoid creating multiple loggers for the same name
_LOGGERS: Dict[str, logging.Logger] = {}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for log levels"""

    def __init__(self, fmt: str = None):
        default_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        super().__init__(fmt or default_fmt)

    def format(self, record):
        # Save original values to restore them later
        levelname = record.levelname
        levelno = record.levelno
        name = record.name
        message = record.getMessage()

        # Apply color to level name based on level
        if levelno in LEVEL_COLORS:
            color = LEVEL_COLORS[levelno]
            record.levelname = f"{color}{levelname}{Colors.RESET}"

        # Format the logger name with blue color
        record.name = f"{Colors.BLUE}{name}{Colors.RESET}"

        # For critical errors, colorize the entire message
        if levelno == logging.CRITICAL:
            record.msg = f"{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD}{message}{Colors.RESET}"

        # Call the original formatter
        result = super().format(record)

        # Restore original values
        record.levelname = levelname
        record.name = name

        return result


def parse_log_level(level: str) -> LogLevel:
    """Convert string log level to LogLevel enum"""
    level_upper = level.upper()
    try:
        return LogLevel[level_upper]
    except KeyError:
        # Fallback to INFO if invalid level
        print(f"Warning: Invalid log level '{level}'. Defaulting to INFO.")
        return LogLevel.INFO


def set_global_log_level(level: Union[LogLevel, str, int]) -> None:
    """
    Set the log level for all existing loggers in the system

    Args:
        level: The new log level - can be LogLevel enum, string name, or int value
    """
    # Convert string level to LogLevel enum if needed
    if isinstance(level, str):
        level = parse_log_level(level)

    # Convert LogLevel enum to int if needed
    if isinstance(level, LogLevel):
        level_value = level.value
    else:
        # Assume it's already an int level
        level_value = level

    # Update root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level_value)

    # Update all handlers on root logger
    for handler in root_logger.handlers:
        handler.setLevel(level_value)

    # Update all cached loggers
    for name, logger in _LOGGERS.items():
        logger.setLevel(level_value)
        # Update all handlers for this logger
        for handler in logger.handlers:
            handler.setLevel(level_value)

    # Create a logger to record this change
    system_logger = get_logger("system", level)
    system_logger.info(f"Global log level set to {getattr(level, 'name', level)}")


def get_logger(
    name: str, level: Union[LogLevel, str, int] = LogLevel.INFO, format_str: Optional[str] = None
) -> logging.Logger:
    """
    Get or create a logger with the specified name and level

    Args:
        name: Logger name (typically the module name)
        level: Logging level - can be LogLevel enum, string name, or int value
        format_str: Optional custom format string

    Returns:
        Configured logger instance
    """
    # Convert string level to LogLevel enum if needed
    if isinstance(level, str):
        level = parse_log_level(level)
    # Convert LogLevel enum to int if needed
    if isinstance(level, LogLevel):
        level_value = level.value
    else:
        # Assume it's already an int level
        level_value = level

    # Check if logger already exists in cache
    if name in _LOGGERS:
        logger = _LOGGERS[name]
        logger.setLevel(level_value)
        return logger

    # Create new logger
    logger = logging.getLogger(name)
    logger.setLevel(level_value)

    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level_value)

    formatter = ColoredFormatter(format_str)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    # Cache for future use
    _LOGGERS[name] = logger

    return logger


def setup_root_logger(level: Union[LogLevel, str, int] = LogLevel.INFO) -> logging.Logger:
    """
    Set up the root logger with colored output

    Args:
        level: Logging level

    Returns:
        Configured root logger
    """
    return get_logger("root", level)
