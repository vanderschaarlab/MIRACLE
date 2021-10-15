# stdlib
from typing import Any

# third party
import numpy as np
from sklearn.impute import KNNImputer


class KNNImputation:
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        self._model = KNNImputer(**kwargs)

    def fit(self, X: np.ndarray, *args: Any, **kwargs: Any) -> "KNNImputation":
        self._model.fit(np.asarray(X), *args, **kwargs)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._model.transform(np.asarray(X))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
