from collections import namedtuple

import causaldag as cd
import cdt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import f1_score

from envs.samplers import D

PRESETS = ["chain", "collider", "fork", "random"]
NOISE_TYPES = ["gaussian", "isotropic-gaussian", "exponential", "gumbel"]
VARIABLE_TYPES = ["gaussian", "non-gaussian", "categorical"]

Data = namedtuple("Data", ["samples", "intervention_node"])


def logmeanexp(A, axis):
    return logsumexp(A, axis=axis) - np.log(A.shape[axis])


def mmd(x, y, kernel):
    # compare kernel MMD paper and code:
    # A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    # http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
    # x shape [n, d] y shape [m, d]
    # n_perm number of bootstrap permutations to get p-value, pass none to not get p-value

    n, d = x.shape
    m, d2 = y.shape
    assert d == d2, "x and y must have same dimensionality"
    k_x = kernel(x, x)
    k_y = kernel(y, y)
    k_xy = kernel(x, y)

    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    mmd = (
        k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    )
    return mmd


class CausalEnvironment(torch.utils.data.Dataset):

    """Base class for generating different graphs and performing ancestral sampling"""

    def __init__(
        self,
        args,
        num_nodes,
        num_edges,
        noise_type,
        num_samples,
        node_range,
        mu_prior=None,
        sigma_prior=None,
        seed=None,
        nonlinear=False,
        binary_nodes=False,
        logger=None,
    ):
        self.args = args
        self.allow_cycles = False
        self.node_range = node_range
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        assert (
            noise_type in NOISE_TYPES
        ), "Noise types must correspond to {} but got {}".format(
            NOISE_TYPES, noise_type
        )
        self.noise_type = noise_type
        self.num_samples = num_samples
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.seed = seed
        self.nonlinear = nonlinear
        self.binary_nodes = binary_nodes
        self.logger = logger

        if seed is not None:
            self.reseed(seed)

        self.init_sampler()

        if nonlinear:
            raise NotImplementedError

        self.sample_weights()
        self.build_graph()

        self.held_out_interventions = []
        self.held_out_nodes = []
        self.held_out_values = []

        for node in range(self.num_nodes):
            # for value in np.linspace(node_range[0], node_range[1], 5):
            for value in [-20, 20]:
                samples = np.zeros(self.num_nodes) + value
                node_binary_v = np.zeros(self.num_nodes)
                node_binary_v[node] = 1.0

                self.held_out_nodes.append(node_binary_v)
                self.held_out_values.append(samples)

                samples = self.intervene(
                    0, 200, node_binary_v, samples, _log=False
                ).samples
                self.held_out_interventions.append(
                    {"node": node, "value": value, "samples": samples}
                )

        self.held_out_data = self.sample(1000).samples

        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(nrows=5)
        # sns.distplot(self.held_out_data[:, 0], ax=ax[0])
        # sns.distplot(self.held_out_data[:, 1], ax=ax[1])
        # sns.distplot(self.held_out_data[:, 2], ax=ax[2])
        # sns.distplot(self.held_out_data[:, 3], ax=ax[3])
        # sns.distplot(self.held_out_data[:, 4], ax=ax[4])
        # fig.savefig('distplot.png')
        # import pdb; pdb.set_trace()

    def reseed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, index):
        raise NotImplementedError

    def build_graph(self):
        """Initilises the adjacency matrix and the weighted adjacency matrix"""

        self.adjacency_matrix = nx.to_numpy_array(self.graph)

        if self.nonlinear:
            self.weighted_adjacency_matrix = None
        else:
            self.weighted_adjacency_matrix = self.adjacency_matrix.copy()
            edge_pointer = 0
            for i in nx.topological_sort(self.graph):
                parents = list(self.graph.predecessors(i))
                if len(parents) == 0:
                    continue
                else:
                    for j in parents:
                        self.weighted_adjacency_matrix[j, i] = self.weights[
                            edge_pointer
                        ]
                        edge_pointer += 1

        print("GT causal graph")
        print(self.adjacency_matrix.astype(np.uint8))

    def init_sampler(self, graph=None):
        if graph is None:
            graph = self.graph

        if self.noise_type.endswith("gaussian"):
            # Identifiable
            if self.noise_type == "isotropic-gaussian":
                self._noise_std = [self.noise_sigma] * self.num_nodes
            elif self.noise_type == "gaussian":
                self._noise_std = np.linspace(0.1, 1.0, self.num_nodes)
            for i in range(self.num_nodes):
                graph.nodes[i]["sampler"] = D(
                    self.rng.normal, loc=0.0, scale=self._noise_std[i]
                )

        elif self.noise_type == "exponential":
            noise_std = [self.noise_sigma] * self.num_nodes
            for i in range(self.num_nodes):
                graph.nodes[i]["sampler"] = D(self.rng.exponential, scale=noise_std[i])

        return graph

    def sample_weights(self):
        """Sample the edge weights"""
        if self.mu_prior is not None:
            # self.weights = torch.distributions.normal.Normal(self.mu_prior, self.sigma_prior).sample([self.num_edges])
            self.weights = D(self.rng.normal, self.mu_prior, self.sigma_prior).sample(
                size=self.num_edges
            )
        else:
            dist = D(self.rng.uniform, -5, 5)
            self.weights = torch.zeros(self.num_edges)
            for k in range(self.num_edges):
                sample = 0.0
                while sample > -0.5 and sample < 0.5:
                    sample = dist.sample(size=1)
                    self.weights[k] = sample

    def sample_linear(
        self, num_samples, graph=None, node=None, values=None, onehot=False
    ):
        """Sample observations given a graph
        num_samples: Scalar
        graph: networkx DiGraph
        node: If intervention is performed, specify which node
        value: value set to node after intervention

        Outputs: Observations [num_samples x num_nodes]
        """

        if graph is None:
            graph = self.graph

        samples = np.zeros((num_samples, self.num_nodes))
        edge_pointer = 0
        for i in nx.topological_sort(graph):
            if onehot and node[i] == 1.0:
                noise = values[i]
            elif not onehot and i == node:
                noise = values
            else:
                noise = self.args.scm_bias + self.graph.nodes[i]["sampler"].sample(
                    num_samples
                )
            parents = list(graph.predecessors(i))
            if len(parents) == 0:
                samples[:, i] = noise
            else:
                curr = 0
                for j in parents:
                    curr += self.weighted_adjacency_matrix[j, i] * samples[:, j]
                    edge_pointer += 1
                curr += noise
                samples[:, i] = curr
        return Data(samples=samples, intervention_node=-1)

    def intervene(self, iteration, num_samples, nodes, values, _log=True):
        """Perform intervention to obtain a mutilated graph"""

        mutated_graph = self.adjacency_matrix.copy()
        if self.binary_nodes:
            mutated_graph[:, nodes.astype(np.bool)] = 0
        else:
            mutated_graph[:, nodes] = 0

        samples = self.sample_linear(
            num_samples,
            self.init_sampler(nx.DiGraph(mutated_graph)),
            nodes,
            values,
            onehot=self.binary_nodes,
        ).samples

        if _log:
            self.logger.log_interventions(iteration, nodes, samples)

        return Data(samples=samples, intervention_node=nodes)

    def sample(self, num_samples):
        _sample = self.sample_linear(num_samples)

        if self.binary_nodes:
            b = np.zeros(self.num_nodes)
            if _sample.intervention_node != -1:
                b[_sample.intervention_node] = 1
            _sample = Data(samples=_sample.samples, intervention_node=b)

        return _sample

    def interventional_likelihood_linear(self, data, intervention):
        graph = self.graph
        logprobs = np.zeros(data.shape[0])
        for i in nx.topological_sort(graph):
            if i == intervention:
                continue
            noise_std = self._noise_std[i]
            parents = list(graph.predecessors(i))
            # TODO: for now we're assuming a zero bias. make it non-zero as well
            bias = self.args.noise_bias
            if len(parents) == 0:
                logprobs += norm.logpdf(data[:, i], bias, noise_std)
            else:
                mean = 0.0
                for j in parents:
                    mean += self.weighted_adjacency_matrix[j, i] * data[:, j]
                # mean += bias
                logprobs += norm.logpdf(data[:, i], mean, noise_std)
        return logprobs

    def interventional_likelihood(self, data, interventions):
        return self.interventional_likelihood_linear(data, interventions)

    def __len__(self):
        return self.num_samples

    def plot_graph(
        self,
        path,
        A=None,
        scores=None,
        dashed_cpdag=True,
        ax=None,
        legend=True,
        save=True,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        if A is None:
            A = self.adjacency_matrix

        try:
            cpdag = cd.DAG().from_amat(amat=A).cpdag().to_amat()[0]
        except:
            cpdag = None

        G = nx.DiGraph(A)
        g = nx.convert_matrix.from_numpy_array(A, create_using=nx.DiGraph)

        pos = {}
        labels = {}
        r = 1
        for i, n in enumerate(range(A.shape[0])):
            theta = np.deg2rad(i * 360 / A.shape[0])
            x, y = r * np.sin(theta), r * np.cos(theta)
            pos[n] = (x, y)
            labels[n] = f"{n+1}"

        edges = G.edges()
        CPDAG_A = np.zeros(A.shape)
        NON_CPDAG_A = np.zeros(A.shape)
        for i, j in edges:
            if cpdag is not None and cpdag[i, j] == cpdag[j, i]:
                CPDAG_A[i][j] = 1
            else:
                NON_CPDAG_A[i][j] = 1

        cmap = plt.cm.plasma

        nodes = nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=g.nodes(),
            node_color=scores,
            node_size=2000,
            edgecolors="black",
            linewidths=5,
            cmap="coolwarm",
        )
        nx.draw_networkx_labels(g, pos, labels, font_color="white")

        nx.draw_networkx_edges(
            nx.convert_matrix.from_numpy_array(NON_CPDAG_A, create_using=nx.DiGraph),
            pos,
            style="solid",
            node_size=1000,
            width=5,
            arrowsize=20,
            connectionstyle="arc3, rad = 0.08",
        )

        collection = nx.draw_networkx_edges(
            nx.convert_matrix.from_numpy_array(CPDAG_A, create_using=nx.DiGraph),
            pos,
            style="dashed",
            node_size=1000,
            width=5,
            arrowsize=20,
            connectionstyle="arc3, rad = 0.08",
        )

        if dashed_cpdag and collection is not None:
            for patch in collection:
                patch.set_linestyle("--")

        ax.set_axis_off()
        if scores is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            cb = plt.colorbar(nodes, cax=cax)
            cb.outline.set_visible(False)

            cax.get_yaxis().labelpad = -60
            cax.set_ylabel("score", rotation=270)

        if legend:
            ax.legend(
                [
                    Line2D([0, 1], [0, 1], linewidth=3, linestyle="-", color="black"),
                    Line2D([0, 1], [0, 1], linewidth=3, linestyle="--", color="black"),
                ],
                [r"$\notin$ CPDAG", r"$\in$ CPDAG"],
                frameon=False,
            )

        if save:
            plt.savefig(path)

    def eshd(self, model, samples, double_for_anticausal=True, force_ensemble=False):
        shds = []

        if model.ensemble or force_ensemble:
            for i in range(len(model.all_graphs)):
                if getattr(model.all_graphs[0], "to_amat", False):
                    G = (model.all_graphs[i].to_amat() != 0).astype(np.uint8)
                else:
                    G = np.array(model.all_graphs[i])
                shds.append(
                    cdt.metrics.SHD(
                        self.adjacency_matrix.copy(),
                        np.array(G),
                        double_for_anticausal=double_for_anticausal,
                    )
                )
        else:
            Gs = model.sample(samples)
            for G in Gs:
                shds.append(
                    cdt.metrics.SHD(
                        self.adjacency_matrix.copy(),
                        np.array(G),
                        double_for_anticausal=double_for_anticausal,
                    )
                )

        return (np.array(shds) * np.exp(model.normalized_log_weights)).sum()

    def get_graphs(self, model, samples=1000, force_ensemble=False):
        if model.ensemble or force_ensemble:
            Gs = []
            for i in range(len(model.all_graphs)):
                if getattr(model.all_graphs[0], "to_amat", False):
                    G = (model.all_graphs[i].to_amat() != 0).astype(np.uint8)
                else:
                    G = np.array(model.all_graphs[i])
                Gs.append(G)
            Gs = np.array(Gs)
        else:
            Gs = np.array(model.sample(samples))

        return Gs

    def f1_score(self, model, samples=1000, force_ensemble=False):
        Gs = self.get_graphs(model, samples, force_ensemble)
        gtGs = self.adjacency_matrix.copy()

        _gtGs = np.broadcast_to(gtGs[None], (Gs.shape[0],) + gtGs.shape)
        f1_scores = []
        for i in range(Gs.shape[0]):
            f1_scores.append(f1_score(_gtGs[i].ravel(), Gs[i].ravel()))
        return (np.array(f1_scores) * np.exp(model.normalized_log_weights)).sum()

    def i_mmd(self, model):
        # get samples from models
        model_samples = model.batch_interventional_samples(
            np.array(self.held_out_nodes)[:, None],
            np.array(self.held_out_values)[:, None],
            200,
            onehot=True,
        )

        # Dags x T x B x N x D
        mmds = []
        for i, (node, value) in enumerate(
            zip(self.held_out_nodes, self.held_out_values)
        ):
            intervention = self.held_out_interventions[i]
            node = intervention["node"]
            value = intervention["value"]
            gt_samples = intervention["samples"]

            _mmds = []
            for dag in range(len(model.all_graphs)):
                _model_samples = model_samples[dag, 0, i]

                # via https://torchdrift.org/notebooks/note_on_mmd.html
                # Gretton et. al. recommend to set the parameter to the median
                # distance between points.

                dists = torch.cdist(
                    torch.Tensor(np.array(_model_samples)),
                    torch.Tensor(np.array(gt_samples)),
                )
                sigma = (dists.median() / 2).item()
                kernel = RBF(length_scale=sigma)
                _mmds.append(mmd(_model_samples, gt_samples, kernel))
            mmds.append((np.array(_mmds) * np.exp(model.normalized_log_weights)).sum())
        return np.array(mmds).mean()
