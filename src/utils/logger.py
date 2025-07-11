"""
Logging Configuration for Experiment Tracking

This module provides standardized logging setup for experiment monitoring,
with both console and file output capabilities.
"""

import sys
import logging
from pathlib import Path


def setup_logger(name="experiment_logger", log_file: Path = None, level=logging.INFO):
    """
    Configure logging for experiment tracking.

    Sets up console and optional file logging with standardized formatting
    for experiment monitoring and debugging.

    Args:
        name: Logger name identifier
        log_file: Optional path for file logging
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"
    )

    # Prevent duplicate handlers
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    # Console
    if not has_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # File
    if log_file and not has_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
