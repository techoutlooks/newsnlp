from typing import Iterable, Union

import numpy as np


class Vocab:
    """
    Iterable corpus dictionary.
    Usage:

        >>> vocab = Vocab(corpus)
        >>> vocab[17]               # 'autorités'
        >>> vocab["autorités"]      # 17

    #TODO: put most frequent words on top? pros? only improves access speed? cf. d2l.ai
    """

    def __init__(self, corpus: Iterable[object]):
        self.tokens = list({str(token) for doc in corpus for token in doc})
        self.word_to_ix = {w: i for (i, w) in enumerate(self.tokens)}

    def __getitem__(self, item: Union[int, str]):
        """ Return a vocab word given its index, and vice-versa. """

        if isinstance(item, int):
            return self.tokens[item]
        if isinstance(item, str):
            return self.word_to_ix[item]

    def __iter__(self):
        for _ in self.tokens:
            yield _

    def __len__(self):
        return len(self.tokens)

    def one_hot(self, items: Iterable[Union[int, str]]):
        """ Get into one-hot vectors from given words (or, their resp. indexes)
        :returns: (|words|, |vocab|)-dimensional np array
        """

        # word list -> ndarray of indexes in vocab;
        # if were passed actual words, get their index instead
        indexes = np.asarray(items)
        if np.issubdtype(indexes.dtype, np.str_):
            indexes = np.asarray([self[item] for item in items])

        # set one-hot position, one word at a time
        X = np.zeros((len(items), len(self)))
        for i, j in enumerate(indexes):
            X[i, j] = 1
        return X

