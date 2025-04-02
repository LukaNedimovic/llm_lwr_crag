import logging
from itertools import zip_longest
from typing import List, Union

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


def log_tc(
    tc_id: int,
    num_tc: int,
    query: str,
    ground_truth_fps: List[str],
    ret_fps: List[str],
    ret_relevant: List[str],
    recall: float,
    gen_ans: str,
) -> None:
    logger.info(f"Test: {tc_id} / {num_tc}")
    logger.info(f"Query: {query}")

    # Compare the ground truth values and the retrieved files
    for gnd, ret in zip_longest(ground_truth_fps, ret_fps, fillvalue=""):
        logger.info(f"{str(gnd):<80} {str(ret)}")

    logger.info(
        (
            f"Common: {len(ret_relevant)} / {len(ground_truth_fps)} "
            f"{recall * 100:.2f}"
        )
    )
    print(f"Answer: {gen_ans}")
