import graphical_models
import networkx as nx
import numpy as np

from .causal_environment import CausalEnvironment
from .utils import expm_np, num_mec


class ErdosRenyi(CausalEnvironment):
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
        args,
        num_nodes,
        exp_edges=1,
        noise_type="isotropic-gaussian",
        noise_sigma=1.0,
        node_range=[-10, 10],
        num_samples=1000,
        mu_prior=2.0,
        sigma_prior=1.0,
        seed=10,
        nonlinear=False,
        binary_nodes=False,
        logger=None,
    ):
        if args.old_er_logic:
            self.noise_sigma = noise_sigma
            p = float(exp_edges) / (num_nodes - 1)
            acyclic = 0
            mmec = 0
            count = 1
            while not (acyclic and mmec):
                if exp_edges <= 2:
                    self.graph = nx.generators.random_graphs.fast_gnp_random_graph(
                        num_nodes, p, directed=True, seed=seed * count
                    )
                else:
                    self.graph = nx.generators.random_graphs.gnp_random_graph(
                        num_nodes, p, directed=True, seed=seed * count
                    )
                acyclic = expm_np(nx.to_numpy_array(self.graph), num_nodes) == 0
                if acyclic:
                    mmec = num_mec(self.graph) >= 2
                count += 1
        else:
            self.noise_sigma = noise_sigma
            p = exp_edges * 2
            print(p)

            n = num_nodes
            dag = np.zeros((n, n))
            ordering = np.arange(n)
            np.random.shuffle(ordering)

            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        continue
                    if np.random.binomial(1, p) == 1:
                        if ordering[i] > ordering[j]:
                            dag[j, i] = 1
                        else:
                            dag[i, j] = 1

            self.graph = nx.DiGraph(dag)

        print(f"MEC SIZE: {num_mec(self.graph)}")

        super().__init__(
            args,
            num_nodes,
            len(self.graph.edges),
            noise_type,
            num_samples,
            node_range=node_range,
            mu_prior=mu_prior,
            sigma_prior=sigma_prior,
            seed=seed,
            nonlinear=nonlinear,
            binary_nodes=binary_nodes,
            logger=logger,
        )

        self.reseed(self.seed)
        self.init_sampler()

        self.dag = graphical_models.DAG.from_nx(self.graph)

        print(f"Expected degree: {np.mean(list(dict(self.graph.in_degree).values()))}")

        self.nodes = self.dag.nodes
        self.arcs = self.dag.arcs

    def __getitem__(self, index):
        return self.samples[index]

    def dag(self):
        return graphical_models.DAG.from_nx(self.graph)
