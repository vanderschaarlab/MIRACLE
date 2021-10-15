# stdlib
import time
from typing import Any, Union

# third party
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import IterativeImputer


class MiceImputation:
    """Imputation plugin for completing missing values using the Multivariate Iterative chained equations and multiple imputations.

    Method:
        Multivariate Iterative chained equations(MICE) methods model each feature with missing values as a function of other features in a round-robin fashion. For each step of the round-robin imputation, we use a BayesianRidge estimator, which does a regularized linear regression.
        The class `sklearn.impute.IterativeImputer` is able to generate multiple imputations of the same incomplete dataset. We can then learn a regression or classification model on different imputations of the same dataset.
        Setting `sample_posterior=True` for the IterativeImputer will randomly draw values to fill each missing value from the Gaussian posterior of the predictions. If each `IterativeImputer` uses a different `random_state`, this results in multiple imputations, each of which can be used to train a predictive model.
        The final result is the average of all the `n_imputation` estimates.

    Args:
        n_imputations: int, default=5i
            number of multiple imputations to perform.
        max_iter: int, default=500
            maximum number of imputation rounds to perform.
        random_state: int, default set to the current time.
            seed of the pseudo random number generator to use.

    """

    def __init__(
        self,
        n_imputations: int = 1,
        max_iter: int = 100,
        random_state: Union[int, None] = None,
    ) -> None:
        if not random_state:
            random_state = int(time.time())

        self._models = []
        for idx in range(n_imputations):
            self._models.append(
                IterativeImputer(
                    max_iter=max_iter,
                    sample_posterior=True,
                    random_state=random_state + idx,
                )
            )

    def fit(self, X: np.ndarray, *args: Any, **kwargs: Any) -> "MiceImputation":
        for model in self._models:
            model.fit(np.asarray(X), *args, **kwargs)

        return self

    def transform(self, X: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        imputations = []
        for model in self._models:
            X_reconstructed = model.transform(X)
            imputations.append(X_reconstructed)

        return np.mean(imputations, axis=0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
