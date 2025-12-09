"""
Tests for pulsarfitpy.utils module.

Tests the configure_logging function for various configurations.
"""

import pytest
import logging
import tempfile
import os

from pathlib import Path
from modules.utils import configure_logging


class TestConfigureLogging:
    """Test suite for configure_logging function."""

    def setup_method(self):
        """Setup before each test - clear existing handlers."""
        logger = logging.getLogger("pulsarfitpy")
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)

    def teardown_method(self):
        """Cleanup after each test."""
        logger = logging.getLogger("pulsarfitpy")
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)

    def test_default_logging_level(self):
        """Test that default logging level is INFO."""
        configure_logging()
        logger = logging.getLogger("pulsarfitpy")
        assert logger.level == logging.INFO

    def test_debug_logging_level(self):
        """Test setting DEBUG logging level."""
        configure_logging("DEBUG")
        logger = logging.getLogger("pulsarfitpy")
        assert logger.level == logging.DEBUG

    def test_warning_logging_level(self):
        """Test setting WARNING logging level."""
        configure_logging("WARNING")
        logger = logging.getLogger("pulsarfitpy")
        assert logger.level == logging.WARNING

    def test_error_logging_level(self):
        """Test setting ERROR logging level."""
        configure_logging("ERROR")
        logger = logging.getLogger("pulsarfitpy")
        assert logger.level == logging.ERROR

    def test_critical_logging_level(self):
        """Test setting CRITICAL logging level."""
        configure_logging("CRITICAL")
        logger = logging.getLogger("pulsarfitpy")
        assert logger.level == logging.CRITICAL

    def test_case_insensitive_level(self):
        """Test that logging level is case insensitive."""
        configure_logging("info")
        logger = logging.getLogger("pulsarfitpy")
        assert logger.level == logging.INFO

        configure_logging("DeBuG")
        assert logger.level == logging.DEBUG

    def test_invalid_level_defaults_to_info(self):
        """Test that invalid logging level defaults to INFO."""
        configure_logging("INVALID_LEVEL")
        logger = logging.getLogger("pulsarfitpy")
        assert logger.level == logging.INFO

    def test_console_handler_added(self):
        """Test that a console handler is added."""
        configure_logging("INFO")
        logger = logging.getLogger("pulsarfitpy")

        # Check that at least one StreamHandler exists
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_file_handler_added(self):
        """Test that file handler is added when log_file is specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            configure_logging("INFO", log_file=log_file)
            logger = logging.getLogger("pulsarfitpy")

            # Check that FileHandler exists
            file_handlers = [
                h for h in logger.handlers if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) == 1

            # Check that log file was created
            assert os.path.exists(log_file)

    def test_custom_format_string(self):
        """Test that custom format string is applied."""
        custom_format = "%(levelname)s - %(message)s"
        configure_logging("INFO", format_string=custom_format)
        logger = logging.getLogger("pulsarfitpy")

        # Check that handlers have the custom format
        for handler in logger.handlers:
            assert handler.formatter._fmt == custom_format

    def test_default_format_string(self):
        """Test that default format string is applied."""
        configure_logging("INFO")
        logger = logging.getLogger("pulsarfitpy")

        default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        for handler in logger.handlers:
            assert handler.formatter._fmt == default_format

    def test_no_duplicate_handlers(self):
        """Test that calling configure_logging multiple times doesn't create duplicate handlers."""
        configure_logging("INFO")
        logger = logging.getLogger("pulsarfitpy")
        handler_count_1 = len(logger.handlers)

        configure_logging("DEBUG")
        handler_count_2 = len(logger.handlers)

        # Should have same number of handlers (old ones cleared)
        assert handler_count_1 == handler_count_2

    def test_logging_actually_works(self, caplog):
        """Test that configured logging actually outputs messages."""
        configure_logging("INFO")
        logger = logging.getLogger("pulsarfitpy")

        with caplog.at_level(logging.INFO, logger="pulsarfitpy"):
            logger.info("Test message")
            assert "Test message" in caplog.text

    def test_file_logging_writes_to_file(self):
        """Test that file logging writes messages to the log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            configure_logging("INFO", log_file=log_file)
            logger = logging.getLogger("pulsarfitpy")

            # Log a test message
            test_message = "This is a test log message"
            logger.info(test_message)

            # Read the log file and check if message is there
            with open(log_file, "r") as f:
                log_contents = f.read()
                assert test_message in log_contents

    def test_both_console_and_file_logging(self):
        """Test that both console and file handlers work together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            configure_logging("INFO", log_file=log_file)
            logger = logging.getLogger("pulsarfitpy")

            # Check we have both types of handlers
            stream_handlers = [
                h
                for h in logger.handlers
                if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
            ]
            file_handlers = [
                h for h in logger.handlers if isinstance(h, logging.FileHandler)
            ]

            assert len(stream_handlers) >= 1
            assert len(file_handlers) == 1

    def test_handler_levels_match_logger_level(self):
        """Test that handler levels match the configured logger level."""
        configure_logging("WARNING")
        logger = logging.getLogger("pulsarfitpy")

        for handler in logger.handlers:
            assert handler.level == logging.WARNING
