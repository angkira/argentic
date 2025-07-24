import logging
import logging.handlers
import sys
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union
import threading
import functools


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

# Lock to ensure thread-safe mutations to global logger caches / configs
_logger_lock = threading.RLock()

# Detect whether we should output ANSI colours – disabled when stdout is not a TTY
# or when NO_COLOR env var is present (https://no-color.org/)
_COLOR_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""


# Global file logging configuration
_FILE_LOGGING_CONFIG: Dict[str, Union[bool, str, int, None]] = {
    "enabled": False,
    "log_dir": None,
    "max_bytes": 10 * 1024 * 1024,  # 10MB per file
    "backup_count": 20,  # Max 20 files
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for log levels"""

    def __init__(self, fmt: Optional[str] = None, enable_color: bool = True):
        # Default format string – keep identical to previous behaviour
        default_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        super().__init__(fmt or default_fmt)

        # Whether colouring should be applied for this formatter instance
        self._enable_color = enable_color and _COLOR_ENABLED

    def format(self, record):
        # Save original values to restore them later
        levelname = record.levelname
        levelno = record.levelno
        name = record.name
        message = record.getMessage()

        if self._enable_color:
            # Apply colour to level name based on level
            if levelno in LEVEL_COLORS:
                color = LEVEL_COLORS[levelno]
                record.levelname = f"{color}{levelname}{Colors.RESET}"

            # Colourise logger name
            record.name = f"{Colors.BLUE}{name}{Colors.RESET}"

            # For critical errors, colourise the entire message
            if levelno == logging.CRITICAL:
                record.msg = f"{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD}{message}{Colors.RESET}"
        else:
            # No colour: keep original values
            pass

        # Call the original formatter
        result = super().format(record)

        # Restore original values
        record.levelname = levelname
        record.name = name

        return result


class PlainFormatter(logging.Formatter):
    """Plain formatter without colors for file output"""

    def __init__(self, fmt: Optional[str] = None):
        default_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        super().__init__(fmt or default_fmt)


def configure_file_logging(
    log_dir: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 20,
    enabled: bool = True,
) -> None:
    """
    Configure global file logging settings.

    Args:
        log_dir: Directory for log files (default: ./logs)
        max_bytes: Maximum size per log file in bytes (default: 10MB)
        backup_count: Maximum number of backup files (default: 20)
        enabled: Whether to enable file logging (default: True)
    """
    global _FILE_LOGGING_CONFIG

    if log_dir is None:
        log_dir = "./logs"

    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    _FILE_LOGGING_CONFIG.update(
        {
            "enabled": enabled,
            "log_dir": str(log_dir),
            "max_bytes": max_bytes,
            "backup_count": backup_count,
        }
    )

    # Update all existing loggers with file handlers
    if enabled:
        for name, logger in _LOGGERS.items():
            _add_file_handler(logger, name)
    else:
        # Remove file handlers if disabling
        for logger in _LOGGERS.values():
            _remove_file_handlers(logger)


def _add_file_handler(logger: logging.Logger, name: str) -> None:
    """Add rotating file handler to a logger."""
    if not _FILE_LOGGING_CONFIG["enabled"] or not _FILE_LOGGING_CONFIG["log_dir"]:
        return

    # Remove existing file handlers to avoid duplicates
    _remove_file_handlers(logger)

    # Create file handler with rotation
    log_dir = str(_FILE_LOGGING_CONFIG["log_dir"])
    max_bytes = int(_FILE_LOGGING_CONFIG["max_bytes"] or 10 * 1024 * 1024)
    backup_count = int(_FILE_LOGGING_CONFIG["backup_count"] or 20)

    log_file = Path(log_dir) / f"{name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )

    # Set level and formatter
    file_handler.setLevel(logger.level)
    file_handler.setFormatter(PlainFormatter())

    # Add to logger
    logger.addHandler(file_handler)


def _remove_file_handlers(logger: logging.Logger) -> None:
    """Remove all file handlers from a logger."""
    handlers_to_remove = [
        h
        for h in logger.handlers
        if isinstance(h, (logging.FileHandler, logging.handlers.RotatingFileHandler))
    ]
    for handler in handlers_to_remove:
        logger.removeHandler(handler)
        handler.close()


# Cache parse_log_level results – avoids repeated dict lookups under heavy logging
@functools.lru_cache(maxsize=16)
def _parse_log_level_cached(level_upper: str) -> LogLevel:
    try:
        return LogLevel[level_upper]
    except KeyError:
        print(f"Warning: Invalid log level '{level_upper}'. Defaulting to INFO.")
        return LogLevel.INFO


def parse_log_level(level: str) -> LogLevel:
    """Convert string log level to LogLevel enum"""
    return _parse_log_level_cached(level.upper())


def set_global_log_level(level: Union[LogLevel, str, int]) -> None:
    """
    Set the global logging level for all loggers and handlers.

    Args:
        level: The desired logging level (LogLevel enum, string name, or int value)
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
    name: str,
    level: Union[LogLevel, str, int] = LogLevel.INFO,
    format_str: Optional[str] = None,
    enable_file_logging: Optional[bool] = None,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    Get or create a logger with the specified name and level

    Args:
        name: Logger name (typically the module name)
        level: Logging level - can be LogLevel enum, string name, or int value
        format_str: Optional custom format string
        enable_file_logging: Override global file logging setting for this logger
        log_dir: Override global log directory for this logger

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

    # Retrieve from cache in a thread-safe manner
    with _logger_lock:
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

    # Force immediate flushing to prevent buffering issues
    console_handler.flush = lambda: sys.stdout.flush()

    formatter = ColoredFormatter(format_str, enable_color=_COLOR_ENABLED)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Add file handler if enabled
    file_logging_enabled = enable_file_logging
    if file_logging_enabled is None:
        file_logging_enabled = _FILE_LOGGING_CONFIG["enabled"]

    if file_logging_enabled:
        if log_dir is not None:
            # Temporarily override global config for this logger
            original_config = _FILE_LOGGING_CONFIG.copy()
            _FILE_LOGGING_CONFIG["log_dir"] = str(log_dir)
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            _add_file_handler(logger, name)

            # Restore original config
            _FILE_LOGGING_CONFIG.update(original_config)
        else:
            _add_file_handler(logger, name)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    # Cache for future use (thread-safe)
    with _logger_lock:
        _LOGGERS[name] = logger

    return logger


def setup_root_logger(
    level: Union[LogLevel, str, int] = LogLevel.INFO,
    enable_file_logging: bool = False,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    Set up the root logger with colored output and optional file logging

    Args:
        level: Logging level
        enable_file_logging: Whether to enable file logging
        log_dir: Directory for log files

    Returns:
        Configured root logger
    """
    if enable_file_logging:
        configure_file_logging(log_dir=log_dir, enabled=True)

    return get_logger("root", level, enable_file_logging=enable_file_logging, log_dir=log_dir)


def get_log_file_info(logger_name: str) -> Optional[Dict[str, Union[str, int, float, bool]]]:
    """
    Get information about the log files for a specific logger.

    Args:
        logger_name: Name of the logger

    Returns:
        Dictionary with log file information or None if no file logging
    """
    if not _FILE_LOGGING_CONFIG["enabled"] or not _FILE_LOGGING_CONFIG["log_dir"]:
        return None

    log_dir = str(_FILE_LOGGING_CONFIG["log_dir"])
    log_file = Path(log_dir) / f"{logger_name}.log"

    info: Dict[str, Union[str, int, float, bool]] = {
        "log_file": str(log_file),
        "log_dir": log_dir,
        "max_bytes": int(_FILE_LOGGING_CONFIG["max_bytes"] or 0),
        "backup_count": int(_FILE_LOGGING_CONFIG["backup_count"] or 0),
        "exists": log_file.exists(),
    }

    if log_file.exists():
        info["size_bytes"] = log_file.stat().st_size
        info["size_mb"] = round(log_file.stat().st_size / (1024 * 1024), 2)

    return info
