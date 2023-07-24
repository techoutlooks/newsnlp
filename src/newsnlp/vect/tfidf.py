from collections import defaultdict, Counter
from functools import reduce

import numpy as np
from ..utils.token_prep import TokenPrep
from ..utils.vocab import Vocab


class TfidfVectorizer:
    """
    TF-IDF implementation. Emulates `sklearn.feature_extraction.text import TfidfVectorizer`
    Minimal usage of SpaCy only for skip_gram wrangling. No use of SpaCy's language vocab,
    nor any custom vectors (eg. w2v). Instead, generates a minimal vector table from
    the input sentences dataset.

    Produces 2D (N,m) vectors, N= #docs, m= #words in vocab,
    the per-doc TF-IDF computed values seating in each vector's components.

    t — term (word), after tokenization
    d — document (set of words)
    m - vocabulary size (in words)
    N — corpus size (in documents)
    corpus — the total document set


    #TODO: num2word, to make `ninety-two` identical to `82`
        use n-gram based statistical modeling? eg. sentencepiece.
    #TODO: handle lang

    Formulas:
    https://courses.engr.illinois.edu/ece448/sp2020/slides/lec38.pdf
    https://lucene.apache.org/core/5_1_0/core/org/apache/lucene/search/similarities/TFIDFSimilarity.html

    Eg. implementations:
    [in R](https://ethen8181.github.io/machine-learning/clustering_old/tf_idf/tf_idf.html)
    [scratch 1](https://www.askpython.com/python/examples/tf-idf-model-from-scratch)
    [scratch 2](https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089)
    """

    def __init__(self, lang=None):

        self._TFIDF = None
        self.tokenizer = TokenPrep(
            lang, lemmatize=True, remove_stop_words=True,
            to_lowercase=True, convert_numbers=True)

    def __call__(self, raw_documents: [str]) -> [str]:

        self.tokenizer(raw_documents)
        self.corpus = list(self.tokenizer.corpus)
        self.vocab = Vocab(self.corpus)
        self.num_docs, self.vocab_size = len(self.corpus), len(self.vocab)
        self.corpus_len = reduce(lambda acc,v: acc+len(v), self.corpus, 0)

        return self

    def similar_to(self, doc_id: int, threshold: float = 0.5, top_n: float = 10):
        """ Top `top_n` docs most similar to `doc_id`,
        having a similarity score of at least `threshold`.
        :returns  [(doc_id, score), ...] as a list of tuples
        """ 
        doc_scores = self.cosine_similarity()[doc_id]
        top_similar = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)
        return list(filter(lambda x: x[1] > threshold, top_similar))[1:top_n]

    def cosine_similarity(self):
        """
        Produces a square (N x N) similarity matrix of cosine scores,
        for all pairwise combinations of vectorized docs. Each doc being
        seated in m-dimensional word vector space.
        :param raw_documents: list of N documents (m-tuples).
        """
        if self._TFIDF is None:  # if not fitted yet
            self.fit()

        # FIXME: IDF with single doc produces div/0
        #    since the term is present in exactly one doc, log(1/1) = 0
        _ = self._TFIDF / np.linalg.norm(self._TFIDF, axis=1).reshape(self.num_docs, 1)
        return np.dot(_, _.T)

    def fit(self):
        self._TFIDF = self.fit_transform()
        return self

    def fit_transform(self):
        """ Learns the corpus' normalised TF-IDF values,
        N = corpus len (in docs), m = vocab size across the N docs.

        Returns a document-term matrix 2D np array (N x m) of:
        - N-dimensional words/features vectors (`vocab_size`) in cols (j ∈ [0, m]),
        - Word vectors having the per-doc tfidf value in each vector component (i ∈ [0, N])
        """
        tfidf_arr = np.zeros((self.num_docs, self.vocab_size))
        for i, doc in enumerate(self.corpus):
            for _, t in enumerate(doc):
                j = self.vocab[t]
                tfidf_arr[i][j] = self.tf_idf(t, i)

        return tfidf_arr

    def tf(self, t: str, doc_id: int):
        """ Term frequency.
            tf(t,d) = count of t in d / total #words in d
        Tweaked as per:
            https://courses.engr.illinois.edu/ece448/sp2020/slides/lec38.pdf
        """
        # return Counter(self.token_counts[t]).get(doc_id, 0) \
        #     / len(self.corpus[doc_id])
        return np.log10(1+Counter(self.token_counts[t]).get(doc_id, 0))
        # return np.log10(1+Counter(self.token_counts[t]).get(doc_id, 0)/len(self.corpus[doc_id]))
        # return 1+ max(0, np.log10(Counter(self.token_counts[t]).get(doc_id, 0)))

    def idf(self, t):
        """ Inverse document frequency.

            idf(t,D) = log( |D| / |{d∈D:t∈d}|+1 )
                     = N / log(df)
                df: #docs comprising t
                df+1, to adjust for words not present in vocab
                      and avoid division by df=0
        """
        df = len(set(self.token_counts[t]))
        return np.log(self.num_docs / (df + 0))

    def tf_idf(self, t: str, doc_id: int):
        """
        tf-idf(t, d) = tf(t, d) * log(idf(t))
            N: len(corpus)
        """
        return self.tf(t, doc_id) * self.idf(t)

    @property
    def token_counts(self):
        """ Word count across entire corpus.
        term -> list of doc ids containing the term
        """
        if not hasattr(self, '_token_counts'):
            self._token_counts = defaultdict(list)
            for doc_id in range(self.num_docs):
                doc = self.corpus[doc_id]
                for t in doc:
                    self._token_counts[str(t)].append(doc_id)

        return self._token_counts
