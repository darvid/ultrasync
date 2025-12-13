"""Logging configuration for galaxybrain using structlog.

Enable debug logging via environment variable:
    GALAXYBRAIN_DEBUG=1 claude

Debug logs are written to .galaxybrain/debug.log in the project directory.
Set GALAXYBRAIN_DEBUG_FILE to override the log file location.

Log levels:
    GALAXYBRAIN_DEBUG=1     → DEBUG level
    GALAXYBRAIN_DEBUG=info  → INFO level (more verbose than default)
    GALAXYBRAIN_DEBUG=warn  → WARNING level only

Logs are automatically rotated at 10MB, keeping 3 backup files.
File logs use JSON format for structured querying.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog
from structlog.typing import EventDict, WrappedLogger

# package-level logger
LOGGER_NAME = "galaxybrain"

# env vars
ENV_DEBUG = "GALAXYBRAIN_DEBUG"
ENV_DEBUG_FILE = "GALAXYBRAIN_DEBUG_FILE"

# log rotation settings
MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 3

# track if we've configured logging
_configured = False


def get_log_level() -> int:
    """Get log level from environment variable."""
    debug_val = os.environ.get(ENV_DEBUG, "").lower()

    if not debug_val:
        return logging.WARNING  # default: only warnings and errors

    if debug_val in ("1", "true", "yes", "debug"):
        return logging.DEBUG
    elif debug_val == "info":
        return logging.INFO
    elif debug_val in ("warn", "warning"):
        return logging.WARNING
    elif debug_val == "error":
        return logging.ERROR
    else:
        return logging.DEBUG  # any truthy value enables debug


def get_log_file(data_dir: Path | None = None) -> Path | None:
    """Get log file path.

    Priority:
    1. GALAXYBRAIN_DEBUG_FILE env var
    2. data_dir/debug.log if data_dir provided
    3. ./.galaxybrain/debug.log
    """
    # explicit env var takes priority
    env_file = os.environ.get(ENV_DEBUG_FILE)
    if env_file:
        return Path(env_file)

    # only write file if debug is enabled
    if not os.environ.get(ENV_DEBUG):
        return None

    # use data_dir or cwd
    if data_dir:
        return data_dir / "debug.log"

    default_dir = Path.cwd() / ".galaxybrain"
    return default_dir / "debug.log"


def add_logger_name(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add logger name to event dict for stdlib compatibility."""
    record = event_dict.get("_record")
    if record is not None and hasattr(record, "name"):
        event_dict["logger"] = record.name
    elif "logger" not in event_dict:
        # try to get name from the bound logger
        if hasattr(logger, "name"):
            event_dict["logger"] = logger.name
        else:
            event_dict["logger"] = LOGGER_NAME
    return event_dict


def drop_internal_keys(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Drop internal keys not needed in output."""
    event_dict.pop("color_message", None)
    event_dict.pop("_record", None)
    event_dict.pop("_from_structlog", None)
    return event_dict


def configure_logging(
    data_dir: Path | None = None,
    force: bool = False,
) -> Any:
    """Configure galaxybrain logging with structlog.

    Call this early in startup to set up logging. Safe to call multiple
    times - subsequent calls are no-ops unless force=True.

    Args:
        data_dir: Directory for storing debug.log (usually .galaxybrain/)
        force: Reconfigure even if already configured

    Returns:
        The configured structlog logger
    """
    global _configured

    if _configured and not force:
        return structlog.get_logger(LOGGER_NAME)

    level = get_log_level()
    log_file = get_log_file(data_dir)

    # shared processors for all outputs
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # configure structlog to use stdlib logging as backend
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # configure stdlib logging to route through structlog
    stdlib_logger = logging.getLogger(LOGGER_NAME)
    stdlib_logger.setLevel(level)
    stdlib_logger.handlers.clear()

    # JSON formatter for file output
    json_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            drop_internal_keys,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )

    # console formatter for stderr (human-readable)
    console_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            drop_internal_keys,
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
    )

    # stderr handler (only if debug enabled)
    if os.environ.get(ENV_DEBUG):
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(level)
        stderr_handler.setFormatter(console_formatter)
        stdlib_logger.addHandler(stderr_handler)

    # rotating file handler with JSON output
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file,
                mode="a",
                maxBytes=MAX_LOG_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(json_formatter)
            stdlib_logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            stdlib_logger.warning(
                "failed to open log file", path=str(log_file), error=str(e)
            )

    # don't propagate to root logger
    stdlib_logger.propagate = False

    _configured = True

    # log startup with structured data
    logger = structlog.get_logger(LOGGER_NAME)
    logger.info(
        "logging configured",
        level=logging.getLevelName(level),
        file=str(log_file) if log_file else None,
        format="json",
        rotation_mb=MAX_LOG_BYTES // 1024 // 1024,
        backup_count=BACKUP_COUNT,
    )

    return logger


def get_logger(name: str | None = None) -> Any:
    """Get a galaxybrain structlog logger.

    Args:
        name: Logger name suffix (e.g., "jit.manager" becomes
              "galaxybrain.jit.manager"). If None, returns root logger.

    Returns:
        Configured structlog logger instance
    """
    if name:
        return structlog.get_logger(f"{LOGGER_NAME}.{name}")
    return structlog.get_logger(LOGGER_NAME)
