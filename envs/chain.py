import numpy as np
import torch
from .causal_environment import CausalEnvironment
import networkx as nx
import graphical_models
from .utils import expm_np, num_mec


class Chain(CausalEnvironment):
    """Generate erdos renyi random graphs using networkx's native random graph builder
    Args:
    num_nodes - Number of Nodes in the graph
    exp_edges - Expected Number of edges in Erdos Renyi graph
    noise_type - Type of exogenous variables
    noise_sigma - Std of the noise type
    num_sampels - number of observations
    mu_prior - prior of weights mean(gaussian)
    sigma_prior - prior of weights sigma (gaussian)
    seed - random seed for data
    """

    def __init__(
        self,
        num_nodes,
        exp_edges=1,
        noise_type="isotropic-gaussian",
        noise_sigma=1.0,
        num_samples=1000,
        mu_prior=2.0,
        sigma_prior=1.0,
        seed=10,
        nonlinear=False,
    ):
        self.noise_sigma = noise_sigma

        A = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes - 1):
            A[i][i + 1] = 1

        self.graph = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
        super().__init__(
            num_nodes,
            len(self.graph.edges),
            noise_type,
            num_samples,
            mu_prior=mu_prior,
            sigma_prior=sigma_prior,
            seed=seed,
            nonlinear=nonlinear,
        )

        self.dag = graphical_models.DAG.from_nx(self.graph)
        self.nodes = self.dag.nodes
        self.arcs = self.dag.arcs

    def __getitem__(self, index):
        return self.samples[index]

    def dag(self):
        return graphical_models.DAG.from_nx(self.graph)
