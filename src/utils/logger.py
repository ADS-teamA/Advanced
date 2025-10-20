"""Logging utilities for the RAG pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional


def clear_log_file(log_file: str) -> None:
    """
    Clear/delete the log file if it exists.
    Call this at the start of pipeline execution to start with fresh logs.

    Args:
        log_file: Path to log file to clear
    """
    log_path = Path(log_file)
    if log_path.exists():
        log_path.unlink()


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
    overwrite: bool = False
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log message format
        log_file: Optional path to log file
        overwrite: If True, overwrite log file instead of appending (default: False)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use 'w' mode to overwrite, 'a' mode to append
        mode = 'w' if overwrite else 'a'
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
