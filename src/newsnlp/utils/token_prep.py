from typing import Iterable
from functools import reduce
import numpy as np

from spacy import Language
from spacy.attrs import LOWER, ENT_TYPE, POS, IS_ALPHA, LEMMA
from spacy.tokens import Doc

import newsnlp.globals as g


# Token attributes and operations we care about
# ignore non-matching
VALID_TOKEN_ATTRS = LOWER, POS, ENT_TYPE, IS_ALPHA, LEMMA
VALID_OPERATIONS = ('token_attrs', 'to_lowercase', 'remove_stop_words', 'lemmatize', 'convert_numbers')


# pick tokenizer config from **params**
extract_tokenizer_config = lambda params, keys: dict([
    (k, (params or {}).get(k)) for k in set(keys).intersection(list(params))])


class TokenPrep:
    """
    Pre-processes a set of documents into a new corpus
    comprising normalized tokens only, using SpaCy's NLP pipeline.

    #TODO: normalize tokens (e.g delete `y'a pas`)
        https://stackoverflow.com/a/46378113
    #TODO: expand contractions (eg. expand `y'a pas -> `il n'y a pas`)
        https://www.kdnuggets.com/2018/08/practitioners-guide-processing-understanding-text-2.html
    #TODO: max_tokens, to impose dataset size limit
    """

    def __init__(self, lang, serialize=True, max_tokens=None, **kwargs):

        self.nlp = g.load_nlp(lang)
        self.serialize = serialize
        if not self.nlp.has_pipe("normalize_tokens"):
            self.nlp.add_pipe("normalize_tokens", config=extract_tokenizer_config(kwargs, VALID_OPERATIONS))

        self.corpus = None

    def __call__(self, raw_documents: [str]):

        # infer corpus
        corpus = self.nlp.pipe(raw_documents)
        if self.serialize:
            corpus = self.to_python(corpus)
        self.corpus = corpus

        return self

    def to_python(self, corpus: Iterable[Doc]) -> Iterable[list]:
        """
        Serialize corpus into a list of sentences (docs),
        each doc being split into bare word strings (tokens).
        Reduces SpaCy objects into Python objects: Doc -> list item, Token -> str.
        """
        return ([token.text for token in doc] for doc in corpus)


class TokenNormalizer:
    """
    SpaCy Pipeline Component that removes junk tokens and applies text transformations.
    Deconstructs a SpaCy Doc into np array of tokens, which is filtered from noise
    (using the token indexes), then packed back into a new SpaCy Doc.

    https://stackoverflow.com/a/52594381
    """

    def __init__(self, token_attrs: dict, to_lowercase: bool,
                 remove_stop_words: bool, lemmatize: bool, convert_numbers: bool):
        self.token_attrs = token_attrs
        self.to_lowercase = to_lowercase
        self.remove_stop_words = remove_stop_words
        self.lemmatize = lemmatize
        self.convert_numbers = convert_numbers

    def __call__(self, doc: Doc) -> Doc:
        # token's text transform pipeline, with cond tests
        lemmatizer = lambda t: t.lemma_ if self.lemmatize else t.text
        to_lowercase = lambda s: s.lower() if self.to_lowercase else s
        compose = lambda *fns: reduce(lambda f, g: lambda t: f(g(t)), fns)

        # destructive, removes junk tokens: stop words, punctuation, symbols, etc.
        # `doc.to_array(token_attrs)` is a projection with sole tokens we care about
        del_token_idxs = [
            i for (i, token) in enumerate(doc) if \
            (self.remove_stop_words and token.is_stop)
            or token.is_punct
            or token.pos_ in ('SYM',)
        ]
        doc_array = doc.to_array(self.token_attrs)
        doc_array = np.delete(doc_array, del_token_idxs, axis=0)
        new_doc = Doc(
            vocab=doc.vocab,
            words=[compose(to_lowercase, lemmatizer)(t)
                   for (i, t) in enumerate(doc) if i not in del_token_idxs]
        )

        new_doc = new_doc.from_array(self.token_attrs, doc_array)
        return new_doc


@Language.factory(
    "normalize_tokens", default_config={
        # caution, will pass params to factory as args, NOT kwargs
        "token_attrs": VALID_TOKEN_ATTRS, "to_lowercase": False,
        "remove_stop_words": False, "lemmatize": False, "convert_numbers": False
    })
def normalize_tokens(nlp: Language, name: str, token_attrs, to_lowercase, remove_stop_words, lemmatize,
                     convert_numbers):
    return TokenNormalizer(token_attrs, to_lowercase, remove_stop_words, lemmatize, convert_numbers)
