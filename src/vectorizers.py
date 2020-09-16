"""
From https://github.com/arosh/BM25Transformer/blob/master/bm25.py
"""
import warnings
from typing import List

import numpy as np
import scipy.sparse as sp
import spacy
import torch
from flair.data import Sentence, Token as FlairToken
from flair.embeddings import TransformerWordEmbeddings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    _document_frequency,
    TfidfVectorizer,
    CountVectorizer,
)
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
from spacy.lang.en import English
from spacy.tokens import Doc, Token
from torch import Tensor


class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b : float, optional (default=0.75)
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """

    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
        return self

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, "dtype") and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        dl = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]
        # In each row, repeat `dl` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(dl), sz)
        # Average document length
        # Scalar value
        avgdl = np.average(dl)
        # Compute BM25 score only for non-zero elements
        data = (
            X.data
            * (self.k1 + 1)
            / (X.data + self.k1 * (1 - self.b + self.b * rep / avgdl))
        )
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, "_idf_diag", "idf vector is not fitted")

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError(
                    "Input has n_features=%d while the model"
                    " has been trained with n_features=%d"
                    % (n_features, expected_n_features)
                )
            # *= doesn't work
            X = X * self._idf_diag

        return X


class MyBM25Transformer(BM25Transformer):
    """
    To be used in sklearn pipeline, transformer.fit()
    needs to be able to accept a "y" argument
    """

    def fit(self, x, y=None):
        super().fit(x)


class BM25Vectorizer(TfidfVectorizer):
    """
    Drop-in, slightly better replacement for TfidfVectorizer
    Best results if text has already gone through stopword removal and lemmatization
    """

    def __init__(self):
        self.vec = make_pipeline(CountVectorizer(binary=True), MyBM25Transformer(),)
        super().__init__()

    def fit(self, raw_documents, y=None):
        return self.vec.fit(raw_documents)

    def transform(self, raw_documents, copy=True):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return self.vec.transform(raw_documents)


class TruncatedSVDVectorizer(TfidfVectorizer):
    def __init__(self, vec: TfidfVectorizer, n_components: int, random_state=42):
        super().__init__()
        self.vec = vec
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)

    def fit(self, texts: List[str], y=None):
        self.vec.fit(texts)
        x = self.vec.transform(texts)
        self.svd.fit(x)

    def transform(self, texts: List[str], copy="deprecated"):
        x = self.vec.transform(texts)
        x = self.svd.transform(x)
        return x


class SpacyVectorizer(TfidfVectorizer):
    def __init__(self, name="en_core_web_lg"):
        super().__init__()
        self.nlp: English = spacy.load(name, disable=["tagger", "ner", "parser"])

    def fit(self, texts: List[str], y=None):
        pass

    @staticmethod
    def doc_to_vecs(doc: Doc) -> List[Tensor]:
        tok: Token
        arrays = [np.array(tok.vector) for tok in doc if tok.has_vector]
        return [torch.from_numpy(a).float() for a in arrays]

    def transform(self, texts: List[str], copy="deprecated") -> List[List[Tensor]]:
        return [self.doc_to_vecs(d) for d in self.nlp.pipe(texts)]


class FlairVectorizer(TfidfVectorizer):
    def __init__(self):
        super().__init__()
        self.embedder = TransformerWordEmbeddings(
            "bert-base-uncased", batch_size=128, layers="-1"
        )

    def fit(self, texts: List[str], y=None):
        pass

    def transform(self, texts: List[str], copy="deprecated") -> List[List[Tensor]]:
        sentences = [Sentence(_) for _ in texts]
        self.embedder.embed(sentences)

        tok: FlairToken
        return [[tok.embedding for tok in s] for s in sentences]
