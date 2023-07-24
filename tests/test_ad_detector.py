
from newsnlp.ad.models.ad_detector import predict_ads
from tests.test_ad import ads_data


def test_predict_ads(ads_data):
    ad_preds = predict_ads(ads_data)
    pass


