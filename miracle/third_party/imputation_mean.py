# stdlib
from typing import Any

# third party
import numpy as np
from sklearn.impute import SimpleImputer


class MeanImputation:
    """Imputation plugin for completing missing values using the Mean Imputation strategy.

    Method:
        The Mean Imputation strategy replaces the missing values using the mean along each column.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        self._model = SimpleImputer(strategy="mean", **kwargs)

    def fit(self, X: np.ndarray, *args: Any, **kwargs: Any) -> "MeanImputation":
        self._model.fit(np.asarray(X), *args, **kwargs)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._model.transform(np.asarray(X))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
