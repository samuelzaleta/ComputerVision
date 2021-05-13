import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AttributerRemover(BaseEstimator, TransformerMixin):
    def __init__(self, remove_hombro_hombro_der=False, remove_hombro_hombro_izq=False, remove_cadera_hombro_der=False, remove_cadera_hombro_izq=False):
        self.remove_hombro_hombro_der = remove_hombro_hombro_der
        self.remove_hombro_hombro_izq = remove_hombro_hombro_izq
        self.remove_cadera_hombro_der = remove_cadera_hombro_der
        self.remove_cadera_hombro_izq = remove_cadera_hombro_izq

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        hombro_hombro_der = X[:, 2]
        cadera_hombro_der = X[:, 3]
        hombro_hombro_izq = X[:, 6]
        cadera_hombro_izq = X[:, 7]

        X = np.delete(X, [2, 3, 6, 7], axis=1)

        if self.remove_hombro_hombro_der:
            X = np.insert(X, -1, hombro_hombro_der, axis=1)
        if self.remove_cadera_hombro_der:
            X = np.insert(X, -1, cadera_hombro_der, axis=1)
        if self.remove_hombro_hombro_izq:
            X = np.insert(X, -1, hombro_hombro_izq, axis=1)
        if self.remove_cadera_hombro_izq:
            X = np.insert(X, -1, cadera_hombro_izq, axis=1)
        return X