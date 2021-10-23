import random
import torch

from ..utils.token_prep import TokenPrep
from ..utils.vocab import Vocab


class DataLoader:
    """
    """

    def __init__(self, context_window_size,
                 randomize_window=False, max_tokens=None):

        self.context_window_size = context_window_size
        self.max_tokens = max_tokens
        self.randomize_window = randomize_window

        self.corpus = None
        self.vocab = None

    def __call__(self, raw_texts):
        self.corpus, self.vocab = self.load_data(raw_texts)
        return self

    def __iter__(self):
        """ Yields minibatches (X, Y) of one-hot word indexes as tensor pairs,
        where (center, context) is expanded as :
        X: center word (repeated),
        Y: context/outside words associated with the center word X.
        """
        for center, context in zip(*self.get_centers_and_contexts()):
            X_ixs, Y_ixs = [], []
            for w_o in context:
                X_ixs += [self.vocab[center]]
                Y_ixs += [self.vocab[w_o]]
            yield torch.LongTensor(X_ixs), torch.LongTensor(Y_ixs)

    def load_data(self, raw_texts):
        """ Load our minibatches, as
        Returns iterable dataset as a 3D tensor
        of shape (num_steps, center_word, vocab_size) """

        tokenizer = TokenPrep(max_tokens=self.max_tokens)(raw_texts)
        corpus = list(tokenizer.corpus)
        vocab = Vocab(corpus)
        return corpus, vocab

    def get_centers_and_contexts(self):
        """Return center words and context words in skip-gram."""

        centers, contexts = [], []
        for line in self.corpus:
            # To form a "center word--context word" pair, each sentence needs to
            # have at least 2 words
            if len(line) < 2:
                continue

            # every word in `line` becomes a center word `i`,
            # as the context window gets centered at `i`
            centers += line
            for i in range(len(line)):
                if self.randomize_window:
                    window_size = random.randint(1, self.context_window_size)
                indices = list(
                    range(max(0, i - self.context_window_size),
                          min(len(line), i + 1 + self.context_window_size))
                )
                # Exclude the center word from the context words
                indices.remove(i)
                contexts.append([line[idx] for idx in indices])

        return centers, contexts




