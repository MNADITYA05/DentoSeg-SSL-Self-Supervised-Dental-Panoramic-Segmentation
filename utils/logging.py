"""
utils/logging.py
─────────────────
Standard Python logger factory.

Returns a logger that writes to both stdout and an optional file,
with a consistent timestamp + level format across all modules.
"""

import logging
import pathlib
import sys
from typing import Optional


def get_logger(
    name: str = "dentoseg",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger.

    Parameters
    ----------
    name : str
        Logger name (shows up in each log line).
    level : int
        Logging level (default INFO).
    log_file : str or None
        If provided, log messages are also written to this file.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_file:
        pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
