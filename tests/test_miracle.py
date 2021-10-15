import sys

import networkx as nx
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.preprocessing import StandardScaler
from utils import binary_sampler, gen_data_nonlinear

import miracle.logger as log
from miracle import MIRACLE

tf.disable_v2_behavior()
np.set_printoptions(suppress=True)
log.add(sink=sys.stderr, level="INFO")


def helper_generate_dummy_data(
    dataset_sz: int = 1000,
    missingness: float = 0.2,
) -> tuple:
    G = nx.DiGraph()
    for i in range(10):
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
    assert ctx.num_inputs == 5 + 5  # input + indicator
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
