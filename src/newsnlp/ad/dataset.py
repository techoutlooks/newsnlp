from typing import Tuple

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from newsnlp.ad import create_ad, Ad
from newsnlp.ad.constants import AD_LOCATOR, ADS_DATASET
from newsnlp.logging import log


__all__ = (
    "Dataset",
    "extract_ad_candidates_from_url",
)

from newsnlp.utils.helpers import load_fixture


def extract_ad_candidates_from_url(url, xpath_or_css=None, raise_exc=True):
    """
    Extract ad candidates from a web url
    An ad corresponds to an anchor tag <A> wrapping an <IMG> tag at least.
    """

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        try:
            page.goto(url)
        except PlaywrightTimeoutError:
            if raise_exc:
                raise
            return []

        ad_candidates = page.locator(xpath_or_css or AD_LOCATOR)
        for link in ad_candidates.all():
            locator = link.locator('img')
            for n in range(locator.count()):
                img = locator.nth(n)
                ad_candidate = create_ad(
                    caption=" ".join(link.all_text_contents()),
                    alt=img.get_attribute('alt'),
                    base_url=url,
                    target_url=link.get_attribute('href'),
                    img_url=img.get_attribute('src'),
                    width=img.get_attribute('width'),
                    height=img.get_attribute('height'),
                    lang='fr'
                )

                yield ad_candidate

        browser.close()


class Dataset:
    """
    Generate an ad dataset (DataFrame) from multiple website urls,
    ready for loading by the loader (unwinds categorical data)
    Exposes a `.save()` method for saving the dataset as a CSV file to disk

    Generates ads predictors similar to [kushmerick99](http://www.sc.ehu.es/ccwbayes/docencia/mmcc/docs/lecturas-clasificacion/abstracts-resumir/kushmerick99learning.pdf)
    but lists values of the categorical predictors in each sample, instead of dummy-encoding them.
    This is to anticipate their conversion into embeddings, suitable for deep learning,
    instead mere statistical learning (cf. our classifier model `newsnlp.classifier.DeepClassifier`).
    """

    def __len__(self):
        return len(self.dataset)

    def __init__(self, ):
        self.dataset = pd.DataFrame()

    def __call__(self, urls):
        """
        Generate the dataset from multiple website urls at once
        Skips urls with timeout while fetching.
        """
        # self.dataset = pd.concat([df for url in urls for df, num_ads in self.exctract_from_url(url)])
        dfs, num_ads = [], 0
        failed_urls = []

        for url in urls:
            df, count = self.exctract_from_url(url)
            dfs += [df]
            num_ads += count
            if not count:
                failed_urls += [url]
        if dfs:
            self.dataset = pd.concat(dfs)

        log.info("done generating {num_ads} ads from {num_ok_urls}/{num_urls} urls\n"
                 "failed urls: {failed_urls}".format(
                    num_ads=num_ads,
                    num_ok_urls=len(urls) - len(failed_urls),
                    num_urls=len(urls),
                    failed_urls=failed_urls))

        return self

    def exctract_from_url(self, url) -> Tuple[pd.DataFrame, int]:
        """
        Extract potential ads from web url.
        An ad is assumed to be <a> tag that embeds an <img> child.

        :param str url: web url to extract candidate ads from
        """

        # fetch unique ads from url
        ad_candidates = extract_ad_candidates_from_url(url, raise_exc=False)
        df = pd.DataFrame(ad_candidates)
        num_ads = len(df)

        log.info(f"extracted {num_ads} ad candidates from {url}")
        return df, num_ads

    def save(self, to=None):
        """ Save dataframe to JSON file """
        # create new index, since unwinding created duplicate rows
        self.dataset.reset_index(inplace=True)
        self.dataset.to_json(to or ADS_DATASET, orient='records')

    def from_json(self, src, save_to=None, ignore_fields=None):

        lang = 'fr'
        args = [k for k in Ad.columns if k not in [
            "aspect_ratio", "local", "is_ad", *list(ignore_fields)]]

        ads_data = load_fixture(src)
        ads_data = map(lambda ad: {**ad, **ad["_raw"]}, ads_data)
        ads_data = map(lambda ad: {k: v for k, v in ad.items() if k in args}, ads_data)
        self.dataset = pd.DataFrame(map(lambda ad: create_ad(**ad, lang=lang), ads_data))

        if save_to:
            self.save(save_to or ADS_DATASET)
