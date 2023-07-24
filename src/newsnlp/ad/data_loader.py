from collections import namedtuple
from functools import reduce

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from newsnlp.ad.ad import Ad, AD_LABEL_COLUMN
from newsnlp.ad.constants import SEED, saved_preprocessor, MIN_PHRASE_COUNT, MAX_PHRASE_LEN
from newsnlp.estimators import get_categorical_counts
from newsnlp.estimators.ordinal_encoder import get_ordinal_encoders
from newsnlp.logging import log
from newsnlp.utils.helpers import save_fixture, load_fixture


__all__ = (
    "ad_predictors",
    "DataLoader",
    "clean_data", "load_preprocessor", "preprocess_data"
)


def categories_to_str(df, col) -> pd.DataFrame:
    """
    Stringify category list per row
    eg. concatenate classes inside each categorical variable
    >>> data = reduce(categories_to_str, Ad.dtypes.categorical, data)
    """
    def to_str(row):
        if isinstance(row[col], (list, tuple)):
            row[col] = "+".join(filter(None, row[col]))
        return row
    return df.apply(to_str, axis=1)


def clean_data(data, min_phrase_count, drop_na=False, drop_rare=False, unwind_cols=None):
    """
    Cleans the row dataset (alters `.dataset`).

    :param pd.DataFrame data: raw dataset
    :param int min_phrase_count: drop rows if occurrence of any class < min_phrase_count
    :param drop_na: drop row iff found at least one NA column. `None` counts as NA
    :param bool drop_rare: drop rows whose categories occur less often than `min_phrase_count`
            only categories for now!
    :param [str] unwind_cols: unwinds columns
    :rtype
    """

    counts = dict(raw=len(data), unwinded=0)

    # unwinding cons: requires reversing `explode()` + mapping predictions (possibly different
    # per unwinded row) back to predictors
    if unwind_cols:
        data = reduce(lambda df, col: df.explode(col), unwind_cols, data)

    if drop_na:
        data.dropna(inplace=True)

    if drop_rare:
        cat_counts = get_categorical_counts(data, Ad.dtypes.categorical)
        for cat, cat_counts in cat_counts.items():
            rare_categories = [k for k, v in cat_counts.items() if v < min_phrase_count]
            rare_rows_df = data[data[cat].isin(rare_categories)]
            data.drop(rare_rows_df.index, inplace=True)

    counts['unwinded'] = len(data)

    log.debug("cleaned dataset: (raw/unwinded) {raw}/{unwinded} ad candidates"
              .format(**counts))

    return data, counts


def load_preprocessor(drop_transformers=None, drop_cols=None):
    """
    Load a pre-pickled preprocessor from file
    :param [str] drop_transformers: name of transformers to remove from pipeline
    :param [str] drop_cols: skip transform for given column names
    """

    p = load_fixture(saved_preprocessor, use_pickle=True)

    if drop_transformers:
        p.transformers = filter(lambda _: _[0] not in drop_transformers, p.transformers)

    if drop_cols:
        for col in drop_cols:
            for i, triplet in enumerate(p.transformers):
                if col in p.transformers[i][2]:
                    p.transformers[i][2].remove(col)
                    if not p.transformers[i][2]:
                        p.transformers[i] = None

    p.transformers = list(filter(None, p.transformers))

    return p


def preprocess_data(data, cont_cols, cat_cols, n_neighbors=5,
                    preprocessor=None, **prep_kw):
    """
    Data preparation in sklearn pipeline, ready for feeding Ad neuralnets.
    Persists (pickle) prep. objects to storage. Performs the following:
    - interpolating missing continuous variables using kNN
    - scaling down (standardize) continuous variables
    - ordinal-encoding categorical variables

    :param pd.DataFrame data: raw data from the ads database
    :param [str] cont_cols: labels wrt `.data` of continuous predictors
    :param [str] cat_cols: labels wrt. `.data` of continuous predictors
    :param int n_neighbors: option for the kNN algorithm
    :param callable or ColumnTransformer preprocessor: column transformer to use,
        instead of the factory created here.
    :param dict prep_kw: kwargs for preprocessor if it is a function
    """

    if not preprocessor:
        
        # Interpolate the numeric columns using KNN and scale using the StandardScaler
        # scaling removes the mean and divides by the std deviation?
        continuous_standardize = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=n_neighbors)),
            ('scaler', StandardScaler())])
    
        # Encode the categorical variables by assigning a number to each of the possibilities.
        # This allows us to use an embedding layer in the PyTorch neural network (NN).
        # eg. pipeline item (innocent): ("ordinal_encode", OrdinalEncoder(), cat_cols),
        # cat_cols_transforms -> triplets (name, transformer, [cat_col]) per categorical var `cat_col`
        # cat_dims -> dimensions of the categorical variables, resp. to `cat_cols` order
        cat_cols_transforms = map(
            lambda _: (f"ordinal_encode_{_[0]}", _[1], [_[0]]),
            get_ordinal_encoders(data, cat_cols))

        preprocessor = ColumnTransformer(transformers=[
            ("continuous_standardize", continuous_standardize, cont_cols),
            *cat_cols_transforms
        ])

        # saved objects, so can re-use the fitted preprocessor to transform
        # raw data from the same distributions, at inference time.
        save_fixture(saved_preprocessor, preprocessor, use_pickle=True)

    if callable(preprocessor):
        preprocessor = preprocessor(**prep_kw)

    # apply transforms and add columns back
    prep_data = preprocessor.fit_transform(data, y=None)
    prep_data = pd.DataFrame(prep_data, columns=cont_cols+cat_cols, index=data.index)

    return prep_data, preprocessor


# Name of dependent variables
ad_predictors = namedtuple(
    'ad_predictors', ['continuous', 'categorical'], defaults=[
        [k for k in Ad.dtypes.continuous if k != Ad.label],
        [k for k in Ad.dtypes.categorical if k != Ad.label]
    ])()


class DataLoader:
    """
    Encodes an Ad dataset loaded from a CSV file into a set of training pairs (X,y)
    """

    def __init__(self, lang, max_phrase_len=None, min_phrase_count=None):

        # hyperparams
        self.max_phrase_len = max_phrase_len or MAX_PHRASE_LEN
        self.min_phrase_count = min_phrase_count or MIN_PHRASE_COUNT

        # dataset props
        self.dataset: pd.DataFrame() = None
        self.label = AD_LABEL_COLUMN
        self.cat_dims = []

        # preprocessed data as X_train, X_test, y_train, y_test
        self.preprocessor = None
        self.data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] = None

    def __len__(self):
        return len(self.dataset)

    def __call__(self, json_file, preprocess=True):
        self.dataset = pd.read_json(json_file)
        self.clean()
        self.preprocess()
        return self

    def clean(self, **kwargs) -> None:
        """
        Cleans the raw un-encoded dataset (alters `.dataset`).

        - drop low occurrences in the dataset
        - discards columns other than `Ad.columns` of interest.
        - doesn't drop NA's by default, NA values are well taken care of by the preprocessor,
            ie., are fit into a class of their own, whilst continuous values are interpolated.

        """
        opts = {'drop_na': False, 'drop_rare': False,
                'unwind_cols': Ad.dtypes.categorical, **(kwargs or {})}
        self.dataset, counts = clean_data(
            self.dataset[Ad.columns], self.min_phrase_count, **opts)

    def preprocess(self, **kwargs):
        """
        Pre-processes a cleaned dataset into train/test sets ready for feeding NN.
        """

        predictors = sum(ad_predictors, [])
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataset[predictors], self.dataset[self.label], random_state=SEED)

        # preprocess the training set
        Xy_train, self.preprocessor = preprocess_data(
            pd.concat([X_train, y_train], axis=1),
            cont_cols=Ad.dtypes.continuous, cat_cols=Ad.dtypes.categorical, **kwargs)
        X_train, y_train = Xy_train[predictors], Xy_train[self.label]

        # preprocess the test set
        Xy_test = self.preprocessor.transform(pd.concat([X_test, y_test], axis=1))
        Xy_test = pd.DataFrame(Xy_test, columns=Ad.columns)
        X_test, y_test = Xy_test[predictors], Xy_test[self.label]

        self.data = X_train, X_test, y_train, y_test

    def save(self, ):
        """ Save post-processed ad data with labels to a csv file """
        pass
