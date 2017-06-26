import numpy as np


class WordVecEmbeddingVectorizer:
    def __init__(self, wordvec):
        self.wordvec = wordvec
        self.dim = wordvec.dim

    def fit_transform(self, X):
        return self.transform(X)

    def transform(self, X):
        return np.array([
            np.mean([self.wordvec[w] for w in words if w in self.wordvec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
