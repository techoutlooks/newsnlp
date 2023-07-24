import pytest

from newsnlp.ad.constants import ADS_DATASET
from newsnlp.ad.data_loader import DataLoader, ad_predictors


# ADS_DATASET = "../data/ads-dataset.json"


def test_load_ads_dataset():
    loader = DataLoader(lang='fr')
    loader(ADS_DATASET)

    predictors = sum(ad_predictors, [])
    X = loader.dataset[predictors]
    y = loader.label

    na_rows_df = X[X.isna().any(axis=1)]
    assert len(na_rows_df) == 0

    print("Predictors shape=", X.shape)
    print("Label counts: is_ad={}, NOT is_ad={}".format(
        len(y[y > 0]), len(y[y == 0])))
