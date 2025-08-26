"""Structured logging configuration for SAMO-DL."""
import logging
import sys
from pathlib import Path

def setup_logging(name: str = "samo-dl", level: str = "INFO") -> logging.Logger:
    """Setup structured logging for the application."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Default logger
logger = setup_logging()
