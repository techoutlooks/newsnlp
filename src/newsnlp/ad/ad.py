import dataclasses
from collections import namedtuple
from typing import Iterable, TypedDict
from urllib.parse import urlparse, urljoin

from newsnlp.ad.constants import MIN_TOKEN_LEN
from newsnlp.utils.helpers import classproperty
from newsnlp.utils.imagesize import get_image_metadata_from_url
from newsnlp.utils.ngram import Ngram


__all__ = (
    "AD_LABEL_COLUMN",
    "create_ad", "ngram_factory", "extract_domain",
    "Ad",
)


AD_LABEL_COLUMN = 'is_ad'


ImageInfo = TypedDict('ImageInfo', {
    'width': int, 'height': int,
    'url': str, 'img_url': str, 'target_url': str,
    'caption': str, 'alt': str
})


@dataclasses.dataclass
class Ad:

    """
    Ad attributes definition & data storage to feed the NN
    Except for the `metadata` fields, all fields are model inputs and used by ML pipelines.
    Defines only ad predictors (inputs), no dependent variable
    """

    # Interval data
    # -----------------------------------------------------------------------
    height: int = 0
    width: int = 0
    aspect_ratio: float = 0

    # Categorical data (non-binary)
    # each of caption, alt, *_url define a subset of one categorical variable
    # -----------------------------------------------------------------------
    caption: [str] = dataclasses.field(default_factory=list)
    alt: [str] = dataclasses.field(default_factory=list)
    base_url: [str] = dataclasses.field(default_factory=list)
    target_url: [str] = dataclasses.field(default_factory=list)
    img_url: [str] = dataclasses.field(default_factory=list)

    # Binary data (is categorical too)
    # -----------------------------------------------------------------------
    # local ie., whether clicking the img ad candidate
    # performs a cross-domain request
    local: bool = False

    # fields starting by and underscore are metadata and
    # skipped by the Loader while generating the (X,y) samples.
    # -----------------------------------------------------------------------
    _raw: ImageInfo = None

    @classproperty
    def label(self):
        return AD_LABEL_COLUMN

    @property
    def inputs(self) -> Iterable[dataclasses.Field]:
        """ Model's input variables as dataclass fields. """
        return filter(lambda f: not f.name.startswith('_'), dataclasses.fields(self))

    @classproperty
    def dtypes(self):
        """ Columns (predictors+output) arranged by their resp.
        statistical scale of measurement """

        return namedtuple('dtypes', ['continuous', 'categorical'], defaults=[

            # continuous variables
            # ---------------------
            ['width', 'height', 'aspect_ratio'],

            # categorical variables
            # ---------------------
            ['local',  # <- binary data, use dummy to save learning complexity?
             'caption', 'alt', 'base_url', 'target_url', 'img_url', 'is_ad']

        ])()

    @classproperty
    def columns(self):
        """ Input columns names for our model """
        return [v for k in self.dtypes for v in k if not v.startswith('_')]


_ngram = None


def ngram_factory(lang='en'):
    """ Initializes the ngrams generator, once for all
    such as a unique instance applies to every `.create_ad()` call.
    """
    global _ngram
    if not _ngram:
        _ngram = Ngram(lang, min_token_len=MIN_TOKEN_LEN)
    return _ngram


def extract_domain(url: str):
    netloc = urlparse(url).netloc
    return ".".join(netloc.split('.')[-2:])


def create_ad(
    caption, alt, base_url, target_url, img_url, width=None, height=None,
    max_phrase_len=2, lang=None
) -> Ad:
    """
    Create a sample from supplied image attributes
    Attempts as much as possible not to download the image.
    :param str caption:
    """
    ngram = ngram_factory(lang)
    ad_candidate = Ad()

    # fix potential relative paths
    img_url = urljoin(base_url, img_url)
    target_url = urljoin(base_url, target_url)

    # geometric image features
    if not (width and height):
        im = get_image_metadata_from_url(img_url, raise_exc=False)
        width, height = im.width, im.height
    else:
        width, height = map(lambda x: str(x).rstrip('px'), (width, height))

    ad_candidate.width, ad_candidate.height = int(float(width)), int(float(height))
    if ad_candidate.width and ad_candidate.height:
        ad_candidate.aspect_ratio = ad_candidate.width / ad_candidate.height

    print(f"ad candidate: {width}x{height}", img_url)

    # is ad candidate img hosted on same sub/domain as the `target_url` of click?
    # iff relative img_url's, assume the base_url domain
    ad_candidate.local = extract_domain(img_url) == extract_domain(target_url)

    for f in filter(lambda x: isinstance(x.type, list), ad_candidate.inputs):
        value = locals()[f.name]

        # prepend the hostname part to url-based features
        # also only generate n-grams of the url path
        if '_url' in f.name:
            getattr(ad_candidate, f.name).append(urlparse(value).hostname)
            value = urlparse(value).path

        # generate n-grams of any list-type feature
        if value:
            for n in range(max_phrase_len):
                ngrams = list(ngram([value]).ngrams(n + 1))
                getattr(ad_candidate, f.name).extend(*ngrams)

    # setup meta fields
    ad_candidate._raw = {
        "img_url": img_url, "width": width, "height": height,
        "base_url": base_url, "target_url": target_url,
        "caption": caption, "alt": alt
    }

    return ad_candidate
