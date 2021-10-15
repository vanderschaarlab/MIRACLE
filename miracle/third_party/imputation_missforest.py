# stdlib
import time
from typing import Any, Union

# third party
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import IterativeImputer


class MissForestImputation:
    """Imputation plugin for completing missing values using the MissForest strategy.

    Method:
        Iterative chained equations(ICE) methods model each feature with missing values as a function of other features in a round-robin fashion. For each step of the round-robin imputation, we use a ExtraTreesRegressor, which fits a number of randomized extra-trees and averages the results.

    Args:
        n_estimators: int, default=10
            The number of trees in the forest.
        max_iter: int, default=500
            maximum number of imputation rounds to perform.
        random_state: int, default set to the current time.
            seed of the pseudo random number generator to use.

    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_iter: int = 100,
        random_state: Union[int, None] = None,
    ) -> None:
        if not random_state:
            random_state = int(time.time())

        estimator_rf = ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=5,
        )
        self._model = IterativeImputer(
            estimator=estimator_rf, random_state=random_state, max_iter=max_iter
        )

    def fit(self, X: np.ndarray, *args: Any, **kwargs: Any) -> "MissForestImputation":
        self._model.fit(np.asarray(X), *args, **kwargs)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._model.transform(np.asarray(X))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
