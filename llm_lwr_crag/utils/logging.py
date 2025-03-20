import logging
from typing import Union

__all__ = ["logger", "setup_logger", "toggle_logger"]
logger = logging.getLogger(__name__)


def setup_logger():
    """
    Set up the logger with a console handler and formatter.
    """
    global logger

    # Avoid adding multiple handlers if setup_logger is called multiple times
    if logger.handlers:
        return

    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)


def toggle_logger(status: Union[bool, int]):
    """
    Toggle logger given the specific status.
    If status is provided as a bool (True / False), it will turn the logger on or off.
    If status is provided as an int, it will shift the status level to it.

    Args:
        status (bool, int): Status to shift the logger to.
    """
    if isinstance(status, bool):
        if status:
            logger.setLevel(logging.INFO)
        else:
            logging.disable(logging.CRITICAL)
    elif isinstance(status, int):
        logger.setLevel(status)
