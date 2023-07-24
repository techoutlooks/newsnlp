"""
Reading:
    - https://dlf.uzh.ch/openbooks/statisticsforlinguists/chapter/n-grams/
    - n-grams and the Markov assumption -> https://youtu.be/-GBgUy6ufUk

"""
import re
from typing import Iterable

import numpy as np
from spacy.tokens import Doc

from newsnlp.utils.token_prep import TokenPrep


def create_ngrams(sentence, n):
    """
    Generate n-grams of unique tokens

    if str, replace all non-alphanumeric characters with spaces
    lowercase, break sentence in the token, remove empty tokens
    :param Doc or str or list: sentence
    """
    sent = str(sentence)
    tokens = set(filter(None, re.sub(r'[^a-zA-Z0-9\s]', ' ', sent.lower()).split(" ")))

    return [" ".join(ngram) for ngram in
            zip(*[list(tokens)[i:] for i in range(n)])]


class Ngram:
    """
    Language-aware n-grams generator based on Spacy
    """

    def __init__(self, *args, min_token_len=2, **kwargs):
        """
        :params min_token_len: drops ngram with to short words (median length of tokens < threshold)
        """
        self.tokenizer = TokenPrep(*args, **kwargs)
        self.min_token_len = min_token_len or 0

    def __call__(self, raw_docs: [str]):
        self.tokenizer(raw_docs)
        return self

    def ngrams(self, n) -> Iterable[Doc]:

        for doc in self.tokenizer.corpus:
            ngrams = create_ngrams(doc, n)
            acceptable_len = (np.median([len(k) for k in doc]) if doc else 0) >= self.min_token_len
            if not self.min_token_len or acceptable_len:
                yield ngrams


