"""
Logging Configuration for DAAM Analysis

Provides centralized logging setup for all DAAM modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup a logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        log_file: Optional file path for log output
        console: Whether to output to console (default: True)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Pre-configured loggers for common modules
def get_daam_logger(name: str = "daam") -> logging.Logger:
    """Get a logger for DAAM analysis modules."""
    return setup_logger(name, level=logging.INFO)
