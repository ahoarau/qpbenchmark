#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Logging with formatting similar to spdlog.

Import ``logging`` from this module to use logging from Python standard library
with formatting similar to spdlog.
"""

import logging
import warnings
from typing import Any, Dict

import tqdm as tqdm_module


def _warning_handler(
    message, category, filename, lineno, file=None, line=None
):
    """Custom warning handler that writes through tqdm.write()."""
    msg = warnings.formatwarning(message, category, filename, lineno, line)
    tqdm_module.tqdm.write(msg.rstrip())


# Redirect warnings through tqdm to avoid interfering with progress bars
warnings.showwarning = _warning_handler


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that writes through tqdm.write().

    This prevents log messages from interfering with tqdm progress bars.
    """

    def __init__(self, level=logging.NOTSET):
        """Initialize the handler with a given level."""
        super().__init__(level)
        self.formatter = SpdlogFormatter()

    def emit(self, record):
        """Emit a log record."""
        try:
            msg = self.format(record)
            tqdm_module.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class SpdlogFormatter(logging.Formatter):
    """Custom logging formatter visually consistent with spdlog."""

    BOLD_RED: str = "\033[31;1m"
    BOLD_WHITE: str = "\033[37;1m"
    BOLD_YELLOW: str = "\033[33;1m"
    GREEN: str = "\033[32m"
    MAGENTA: str = "\033[35m"
    ON_RED: str = "\033[41m"
    RESET: str = "\033[0m"

    LEVEL_FORMAT: Dict[Any, str] = {
        logging.CRITICAL: f"[{ON_RED}{BOLD_WHITE}critical{RESET}]",
        logging.DEBUG: f"[{MAGENTA}debug{RESET}]",
        logging.ERROR: f"[{BOLD_RED}error{RESET}]",
        logging.INFO: f"[{GREEN}info{RESET}]",
        logging.WARNING: f"[{BOLD_YELLOW}warning{RESET}]",
    }

    def format(self, record):
        """Format for a logging record.

        Args:
            record: Record to format.
        """
        custom_format = (
            "[%(asctime)s] "
            + self.LEVEL_FORMAT.get(record.levelno, "[???]")
            + " %(message)s (%(filename)s:%(lineno)d)"
        )
        formatter = logging.Formatter(custom_format)
        return formatter.format(record)


# Initialize logger with TqdmLoggingHandler for clean progress bar output
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid duplicate logs
logger.handlers.clear()

# Add our tqdm-aware handler
handler = TqdmLoggingHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


__all__ = [
    "logging",
    "TqdmLoggingHandler",
]
