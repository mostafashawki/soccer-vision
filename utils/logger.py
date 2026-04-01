"""Structured JSON logger for soccer-vision pipeline."""

import logging
import json
import sys
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines for machine-readable output."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        return json.dumps(log_entry)


def get_logger(
    name: str = "soccer-vision",
    level: str = "INFO",
    json_output: bool = True,
) -> logging.Logger:
    """Create a configured logger instance.

    Args:
        name: Logger name (typically module name).
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, outputs JSON lines. If False, uses human-readable format.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)

    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def log_with_data(
    logger: logging.Logger,
    level: str,
    message: str,
    data: Optional[dict] = None,
) -> None:
    """Log a message with optional structured data attached.

    Args:
        logger: Logger instance.
        level: Log level string.
        message: Log message.
        data: Optional dictionary of extra data to include in the log entry.
    """
    log_method = getattr(logger, level.lower(), logger.info)
    if data:
        extra_record = logging.LogRecord(
            name=logger.name,
            level=getattr(logging, level.upper(), logging.INFO),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        extra_record.extra_data = data
        logger.handle(extra_record)
    else:
        log_method(message)
