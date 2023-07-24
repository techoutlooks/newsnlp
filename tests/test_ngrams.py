
from newsnlp.utils.ngram import Ngram


def test_ngrams(n=2, min_token_len=2):

    # example corpus
    ok_sent = 'my father will succeed'
    fail_sent = 't nt'
    corpus = [ok_sent, fail_sent]

    ng = Ngram('fr', min_token_len)(corpus)
    result = list(ng.ngrams(n))
    print(result)
