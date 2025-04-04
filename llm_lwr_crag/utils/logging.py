import csv
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
    gen_ans: Union[str, None],
) -> None:
    """
    Log the outcome of a single test case.

    Args:
        tc_id (int): Test case ID.
        num_tc (int): Total number of test cases.
        query (str): User query.
        ground_truth_fps (List[str]): List of ground truth files from the
            evaluation dataset.
        ret_fps (List[str]): List of retrieved file paths.
        ret_relevant (List[str]): List of file paths retrieved, that match the
            ground truth values.
        recall (float): Recall@K metric
        gen_ans (Union[None, str]): If applicable, LLM generated answer.

    Returns:
        None
    """
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


def log_res(
    log_path: str,
    log_dict: dict,
):
    # Format query augmentation logging
    eval = log_dict["eval"]
    if eval is None:
        aug_query_log = ""
    else:
        aug_query = eval.get("augment_query", None)
        if aug_query:
            aug_query_log = f"provider={aug_query.get('provider')}, llm={aug_query.get('model_name', None)}"
        else:
            aug_query_log = "None"
    log_dict["eval"] = aug_query_log

    # Format metadata logging
    metadata = log_dict["metadata"]
    metadata_log = ""
    if metadata:
        llm_summary = metadata.get("llm_summary", None)
        if llm_summary:
            metadata_log = (
                f"{metadata.get('list', 'no metadata')} | "
                f"llm_summary: (provider={llm_summary.get('provider')}, llm={llm_summary.get('model_name', None)})"  # noqa: E501
            )
    else:
        metadata_log = "None"
    log_dict["metadata"] = metadata_log

    log_entry = log_dict.values()
    with open(log_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(log_entry)
