import pandas as pd
from box import Box
from utils.logging import logger

from .metadata import augment_query


def preprocess_eval(eval_df: pd.DataFrame, metadata_args: Box) -> pd.DataFrame:
    if metadata_args is None:
        return
    if metadata_args.augment_query:
        logger.info("Augmenting queries...")
        eval_df["question"] = eval_df["question"].apply(
            lambda query: augment_query(query, metadata_args)
        )
        logger.info("Finished augmenting queries...")
