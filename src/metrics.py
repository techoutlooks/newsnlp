"""
Doc similarity calculation routines.
Assumes functions inputs to be document-term matrices,
ie. np arrays (N,m) were:
    m - vocabulary size (in words)
    N — corpus size (in documents)
"""
import numpy as np


def ranking_score_similarity(self):
    """
    Compares the document-wise sum of tfidf values matching the query
    :return:
    """
    pass


def pairwise_cosine_similarity(A, B):
    """
    Produces a rectangular (N_a x N_b) similarity matrix for all (N_a x N_b) pairwise
    combinations of documents, from two document sets (resp., N_a x m, N_b x m) seated
    in m-dimensional word vector space.
    """
    return np.dot(A, B.T) / (normalize(A) * normalize(B))


def normalize(A):
    """
    Document or corpus n
    :param A:
    :return:
    """
    N, m = A.shape
    return np.linalg.norm(A, axis=1).reshape(N, 1)


