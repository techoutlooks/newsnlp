from urllib.parse import urljoin

import pytest

from newsnlp.ad.dataset import extract_ad_candidates_from_url, Dataset
from newsnlp.ad import extract_domain
from newsnlp.ad.constants import ADS_DATASET, data_dir
from newsnlp.utils.helpers import load_fixture, save_fixture


@pytest.fixture
def urls():
    """
    Top ranking online newspapers in supported countries
    https://www.semrush.com/trending-websites/sn/newspapers
    https://library.stanford.edu/africa-south-sahara/browse-country
    https://www.allyoucanread.com/
    """
    return [

        # Senegal
        "https://www.igfm.sn/",
        "https://www.seneweb.com/",
        "https://www.dakaractu.com/",
        "https://www.rewmi.com/",
        "https://lesoleil.sn/"

        # Guinea
        "https://guineenews.org/"
        "https://www.africaguinee.com/"
        "https://guineematin.com/",
        "https://mosaiqueguinee.com/",
        "https://lerevelateur224.com/",

        # Cote d'Ivoire
        "http://abidjan.net/",
        "https://www.fratmat.info/",
        "https://www.aip.ci/",
        "https://www.linfodrome.com/",
        "https://gbich.net/"
    ]


def test_extract_ad_candidates_from_url(urls):
    ad_candidates = list(extract_ad_candidates_from_url(urls[0]))
    assert ad_candidates


def test_create_ads_dataset(urls):
    creator = Dataset()
    creator = creator(urls)
    assert creator
    creator.save(ADS_DATASET)


def test_create_dataset_from_json():
    creator = Dataset()
    ignore_fields = ("base_url", "target_url", "img_url", )
    creator.from_json(src=ADS_DATASET, save_to=f"{data_dir}/ads-new.json",
                      ignore_fields=ignore_fields)


def test_fix_ad_local_field():

    ads_data = load_fixture(ADS_DATASET)

    for i, ad in enumerate(ads_data):

        base_url = ad["_raw"]["base_url"]
        img_url = urljoin(base_url, ad["_raw"]["img_url"])
        target_url = urljoin(base_url, ad["_raw"]["target_url"])
        is_local = extract_domain(img_url) == extract_domain(target_url)

        # assert ad["local"] == is_local, f"Not equal, ad #{i} requires fix"
        ad["local"] = extract_domain(img_url) == extract_domain(target_url)

    save_fixture(ADS_DATASET, ads_data)

