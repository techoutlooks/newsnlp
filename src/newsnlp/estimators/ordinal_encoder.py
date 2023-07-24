from collections import OrderedDict, Counter
from typing import Tuple, List

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


__all__ = ("get_categorical_counts", "get_ordinal_encoders")


def get_categorical_counts(df: pd.DataFrame, cat_cols) -> OrderedDict[str, Counter]:
    """ Count the occurrences of values by category """

    # df['alt'].value_counts()
    return OrderedDict({k: Counter(df[k]) for k in cat_cols})


def get_ordinal_encoders(df: pd.DataFrame, cat_cols, **kwargs):
    """
    Initializes OrdinalEncoder's that also ordinal-encode unknown and missing classes.

    Yields ordinal encoders calibrated for given categories (respectively),
    assigning new classes to unknown (unseen during training) and missing (NA) values.
    Increases the number of classes by +2 from the number of classes present in the df.

    :return: Yields `len(cat_cols)` initialized encoders, one per each categorical variable.
    """

    cat_counts = get_categorical_counts(df, cat_cols)
    for cat, counts in cat_counts.items():

        num_classes = len(counts.values())
        enc = OrdinalEncoder(max_categories=num_classes,
            handle_unknown='use_encoded_value', unknown_value=num_classes,
            encoded_missing_value=num_classes+1, **kwargs)

        yield cat, enc

