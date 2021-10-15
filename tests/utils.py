import random
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import torch

np.set_printoptions(suppress=True)
# This function generates data according to a DAG provided in list_vertex and list_edges with mean and variance as input
# It will apply a perturbation at each node provided in perturb.


def gen_data_nonlinear(
    G: Any,
    base_mean: float = 0,
    base_var: float = 0.3,
    mean: float = 0,
    var: float = 1,
    SIZE: int = 10000,
    err_type: str = "normal",
    perturb: list = [],
    sigmoid: bool = True,
    expon: float = 1.1,
) -> pd.DataFrame:
    list_edges = G.edges()
    list_vertex = G.nodes()

    order = []
    for ts in nx.algorithms.dag.topological_sort(G):
        order.append(ts)

    g = []
    for v in list_vertex:
        if v in perturb:
            g.append(np.random.normal(mean, var, SIZE))
            print("perturbing ", v, "with mean var = ", mean, var)
        else:
            if err_type == "gumbel":
                g.append(np.random.gumbel(base_mean, base_var, SIZE))
            else:
                g.append(np.random.normal(base_mean, base_var, SIZE))

    for o in order:
        for edge in list_edges:
            if o == edge[1]:  # if there is an edge into this node
                if sigmoid:
                    g[edge[1]] += 1 / 1 + np.exp(-g[edge[0]])
                else:
                    g[edge[1]] += g[edge[0]] ** 2
                    # if edge[0] % 2 == 0:
                    #    g[edge[1]] += np.power(np.abs(g[edge[0]]), expon)
                    # else:
                    #    g[edge[1]] += np.log(np.abs(g[edge[0]])) * edge[0]
                    # g[edge[1]] +=  g[edge[0]]
    g = np.swapaxes(g, 0, 1)

    return pd.DataFrame(g, columns=list(map(str, list_vertex)))


def handler(signal_received: Any, frame: Any) -> None:
    # Handle any cleanup here
    print("SIGINT or CTRL-C detected. Exiting gracefully")
    exit(0)


def binary_sampler(p: float, rows: int, cols: int) -> np.ndarray:
    """Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    """
    unif_random_matrix = np.random.uniform(0.0, 1.0, size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix


def enable_reproducible_results(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    tf.random.set_random_seed(seed)
