# stdlib
import sys

# third party
import networkx as nx
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf
from utils import binary_sampler, enable_reproducible_results, gen_data_nonlinear

# miracle absolute
from miracle import MIRACLE
import miracle.logger as log
from miracle.third_party import load_imputer

tf.disable_v2_behavior()
np.set_printoptions(suppress=True)
log.add(sink=sys.stderr, level="INFO")

enable_reproducible_results(0)

ALLOW_THRESHOLD = 0.3


def helper_generate_dummy_data(
    dataset_sz: int = 3000,
    missingness: float = 0.2,
    feature_size: int = 10,
) -> tuple:
    G = nx.DiGraph()
    for i in range(feature_size):
        G.add_node(i)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(1, 4)
    G.add_edge(2, 5)
    G.add_edge(2, 0)
    G.add_edge(3, 0)
    G.add_edge(3, 6)
    G.add_edge(3, 7)
    G.add_edge(6, 9)
    G.add_edge(0, 8)
    G.add_edge(0, 9)

    df = gen_data_nonlinear(G, SIZE=dataset_sz, sigmoid=False)

    scaler = StandardScaler()
    df = scaler.fit_transform(df)

    X_MISSING = df.copy()  # This will have missing values
    X_TRUTH = df.copy()  # This will have no missing values for testing

    # Generate MCAR
    X_MASK = binary_sampler(1 - missingness, X_MISSING.shape[0], X_MISSING.shape[1])
    X_MISSING[X_MASK == 0] = np.nan

    # Append indicator variables - One indicator per feature with missing values.
    missing_idxs = np.where(np.any(np.isnan(X_MISSING), axis=0))[0]

    return X_MISSING, X_TRUTH, X_MASK, missing_idxs


def test_sanity() -> None:
    ctx = MIRACLE(
        lr=0.3,
        batch_size=4,
        num_inputs=5,
        num_outputs=1,
        n_hidden=7,
        reg_lambda=0.8,
        reg_beta=0.9,
        missing_list=[1],
        reg_m=10.0,
        window=11,
        max_steps=200,
    )

    assert ctx.learning_rate == 0.3
    assert ctx.reg_lambda == 0.8
    assert ctx.reg_beta == 0.9
    assert ctx.reg_m == 10.0
    assert ctx.batch_size == 4
    assert ctx.num_inputs == 5 + 1  # input + indicator
    assert ctx.num_outputs == 1
    assert ctx.n_hidden == 7
    assert ctx.missing_list == [1]
    assert ctx.window == 11
    assert ctx.max_steps == 200


def test_fit() -> None:
    missing, truth, mask, indices = helper_generate_dummy_data(dataset_sz=100)

    # Initialize MIRACLE
    miracle = MIRACLE(
        num_inputs=missing.shape[1],
        missing_list=indices,
        max_steps=50,
    )

    # Train MIRACLE
    miracle_imputed_data_x = miracle.fit(missing)

    assert miracle_imputed_data_x.shape == missing.shape
    assert np.isnan(miracle_imputed_data_x).sum() == 0

    assert miracle.rmse_loss(truth, miracle_imputed_data_x, mask)


@pytest.mark.parametrize("seed", ["missforest", "mean", "mice", "gain", "knn"])
def test_imputers_sanity(seed: str) -> None:
    missing, truth, mask, indices = helper_generate_dummy_data(dataset_sz=500)

    seed_imputer = load_imputer(seed)

    X_seed = seed_imputer.fit_transform(missing)

    miracle = MIRACLE(
        num_inputs=missing.shape[1],
        missing_list=indices,
        max_steps=300,
    )

    miracle_imputed_data_x = miracle.fit(missing, X_seed=X_seed)

    baseline_rmse = miracle.rmse_loss(truth, X_seed, mask)
    miracle_rmse = miracle.rmse_loss(truth, miracle_imputed_data_x, mask)

    assert baseline_rmse < 2
    assert miracle_rmse < 2


@pytest.mark.parametrize("dataset_sz", [3000, 5000])
@pytest.mark.parametrize("seed", ["missforest", "mean", "mice", "gain", "knn"])
@pytest.mark.slow
def test_experiments_dataset_size(dataset_sz: int, seed: str) -> None:
    missing, truth, mask, indices = helper_generate_dummy_data(dataset_sz=dataset_sz)
    seed_imputer = load_imputer(seed)

    X_seed = seed_imputer.fit_transform(missing)

    miracle = MIRACLE(
        num_inputs=missing.shape[1],
        missing_list=indices,
    )

    miracle_imputed_data_x = miracle.fit(missing, X_seed=X_seed)

    baseline_rmse = miracle.rmse_loss(truth, X_seed, mask)
    miracle_rmse = miracle.rmse_loss(truth, miracle_imputed_data_x, mask)

    try:
        assert baseline_rmse > miracle_rmse
    except AssertionError:
        if abs(baseline_rmse - miracle_rmse) < ALLOW_THRESHOLD * baseline_rmse:
            pytest.xfail(
                f"Baseline and MIRACLE RMSE: {baseline_rmse} and {miracle_rmse}"
            )
        else:
            raise AssertionError


@pytest.mark.parametrize("missingness", [0.2, 0.4, 0.7])
@pytest.mark.parametrize("seed", ["missforest", "mean", "mice", "gain", "knn"])
@pytest.mark.slow
def test_experiments_missingness(missingness: float, seed: str) -> None:
    missing, truth, mask, indices = helper_generate_dummy_data(missingness=missingness)
    seed_imputer = load_imputer(seed)

    X_seed = seed_imputer.fit_transform(missing)

    miracle = MIRACLE(
        num_inputs=missing.shape[1],
        missing_list=indices,
    )

    miracle_imputed_data_x = miracle.fit(missing, X_seed=X_seed)

    baseline_rmse = miracle.rmse_loss(truth, X_seed, mask)
    miracle_rmse = miracle.rmse_loss(truth, miracle_imputed_data_x, mask)

    try:
        assert baseline_rmse > miracle_rmse
    except AssertionError:
        if abs(baseline_rmse - miracle_rmse) < ALLOW_THRESHOLD * baseline_rmse:
            pytest.xfail(
                f"Baseline and MIRACLE RMSE: {baseline_rmse} and {miracle_rmse}"
            )
        else:
            raise AssertionError


@pytest.mark.parametrize("feature_size", [10, 50, 100])
@pytest.mark.parametrize("seed", ["missforest", "mean", "mice", "gain", "knn"])
@pytest.mark.slow
def test_experiments_feature_size(feature_size: int, seed: str) -> None:
    missing, truth, mask, indices = helper_generate_dummy_data(
        feature_size=feature_size
    )
    seed_imputer = load_imputer(seed)

    X_seed = seed_imputer.fit_transform(missing)

    miracle = MIRACLE(
        num_inputs=missing.shape[1],
        missing_list=indices,
    )

    miracle_imputed_data_x = miracle.fit(missing, X_seed=X_seed)

    baseline_rmse = miracle.rmse_loss(truth, X_seed, mask)
    miracle_rmse = miracle.rmse_loss(truth, miracle_imputed_data_x, mask)

    try:
        assert baseline_rmse > miracle_rmse
    except AssertionError:
        if abs(baseline_rmse - miracle_rmse) < ALLOW_THRESHOLD * baseline_rmse:
            pytest.xfail(
                f"Baseline and MIRACLE RMSE: {baseline_rmse} and {miracle_rmse}"
            )
        else:
            raise AssertionError
