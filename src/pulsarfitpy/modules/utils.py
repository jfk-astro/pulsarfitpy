"""
Utility functions for pulsarfitpy.

This module provides helper functions including logging configuration.
"""

import logging


def configure_logging(level="INFO", format_string=None, log_file=None):
    """
    Configure logging for pulsarfitpy.

    Parameters
    ----------
    level : str, optional
        Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        Default is 'INFO'.
    format_string : str, optional
        Custom format string for log messages.
        Default is '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file : str, optional
        Path to log file. If None, logs to console only.

    Examples
    --------
    >>> from pulsarfitpy import configure_logging
    >>> # Show all info messages
    >>> configure_logging('INFO')
    >>>
    >>> # Only show warnings and errors
    >>> configure_logging('WARNING')
    >>>
    >>> # Show debug info and save to file
    >>> configure_logging('DEBUG', log_file='training.log')
    >>>
    >>> # Suppress all training output
    >>> configure_logging('ERROR')
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_level = level_map.get(level.upper(), logging.INFO)

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get the pulsarfitpy logger
    pulsarfitpy_logger = logging.getLogger("pulsarfitpy")
    pulsarfitpy_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    pulsarfitpy_logger.handlers.clear()

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    pulsarfitpy_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        pulsarfitpy_logger.addHandler(file_handler)
        pulsarfitpy_logger.info(f"Logging to file: {log_file}")

    pulsarfitpy_logger.info(f"Logging configured at level: {level.upper()}")
