import os
import pickle

import causaldag as cd
import graphical_models
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats.norm import logpdf
from scipy.special import logsumexp

from models.dag_bootstrap_lib import utils

from .posterior_model import PosteriorModel


def logmeanexp(A, axis):
    return logsumexp(A, axis=axis) - np.log(A.shape[axis])


class DagBootstrap(PosteriorModel):
    def __init__(self, env, args, covariance_matrix=None):
        super().__init__()
        # todo: get from config

        self.ensemble = True

        self.args = args
        self.num_nodes = args.num_nodes
        self.num_bootstraps = args.num_particles
        self.group_interventions = args.group_interventions
        self.seed = args.seed

        self.dags = None

        a = graphical_models.DAG.from_nx(env.graph)
        skeleton = a.cpdag()  ##Find the skeleton

        # sample at most self.num_bootstraps from groudntruth MEC
        mec_dags = skeleton.all_dags()  # Find
        self.graph_nodes = a.nodes

        if len(mec_dags) > self.num_bootstraps:
            self.mec_dags = np.array(list(mec_dags))[
                np.random.choice(
                    range(len(mec_dags)), self.num_bootstraps, replace=False
                )
            ].tolist()
        else:
            self.mec_dags = np.array(list(mec_dags)).tolist()

        # precision matrix is already known
        self.covariance_matrix = covariance_matrix

    def update(self, data, bic_weighting=False, cleanup=True):
        assert self.covariance_matrix.shape[0] == self.num_nodes

        cov_mat = self.covariance_matrix

        if (not self.args.reuse_posterior_samples) or (self.dags is None):
            tmp_path = utils.run_gies_boot(
                data,
                self.num_bootstraps,
                group_interventions=self.group_interventions,
                maintain_int_dist=True,
            )
            self.amats, dags = utils._load_dags(tmp_path=tmp_path, delete=cleanup)
            self.dags = [utils.cov2dag(cov_mat, dag) for dag in dags]
            self.all_graphs = self.dags
        else:
            print("reusing posterior samples from previous update")

        if self.args.include_gt_mec:
            for mec_graph in self.mec_dags:
                self.dags.append(
                    utils.cov2dag(
                        cov_mat,
                        graphical_models.DAG(arcs=mec_graph, nodes=self.graph_nodes),
                    )
                )

            self.functional_matrix = np.ones([len(self.dags), self.num_nodes])

        self.log_weights = np.log(np.ones(len(self.dags)))
        self.normalized_log_weights = np.log(np.ones(len(self.dags)) / len(self.dags))

        if self.args.weighted_posterior:
            weights = (
                self.batch_likelihood(
                    data.nodes[:, None],
                    data.samples[None, None, :, None],
                    onehot=True,
                    weights=False,
                )
                .sum(0)
                .ravel()
            )

            if bic_weighting:
                A = np.array([g.to_amat() for g in self.all_graphs])
                weights -= A.sum((1, 2)) * np.log(data.samples.shape[0]) / 2

            tau = 1
            self.log_weights = weights / tau
            self.normalized_log_weights = weights / tau - logsumexp(weights / tau)

    def batch_interventional_samples(
        self, nodes, values, n_samples, deterministic=False, onehot=False, precise=True
    ):
        # Dags x T x B x N x D

        B, T, D = values.shape
        ndags = len(self.all_graphs)

        N = self.all_graphs[0].to_amat().shape[0]

        A = np.array([g.to_amat() for g in self.all_graphs])
        topological_sorts = np.array([g.topological_sort() for g in self.all_graphs])

        P = np.eye(N)[topological_sorts]  # permutation matrix

        tri = P @ A @ P.transpose([0, 2, 1])

        variances = np.array([g._variances for g in self.all_graphs]).T
        biases = np.array([g._biases for g in self.all_graphs]).T

        if deterministic:
            noises = np.zeros((T, B, N, ndags, n_samples))
        else:
            noises = np.random.randn(T, B, N, ndags, n_samples)
        # noise = np.random.normal(size=[nsamples, len(self._nodes)])
        # noise = noise * np.array(self._variances) ** .5 + self._biases

        noises = (
            noises * variances[None, None, :, :, None] ** 0.5
            + biases[None, None, :, :, None]
        )

        if onehot:
            # transpose from B,T,N to T,B,N
            nodes_one_hot = nodes.transpose([1, 0, 2])
        else:
            nodes_one_hot = jnp.zeros((B * T, N))
            nodes_one_hot = nodes_one_hot.at[jnp.arange(B * T), nodes.ravel()].set(1)
            nodes_one_hot = nodes_one_hot.reshape(B, T, N).transpose([1, 0, 2])

        noises = noises.transpose([3, 0, 1, 4, 2])

        if precise:
            # sample = nodes_one_hot[None, :, :, None, :]
            sample = (nodes_one_hot * values.transpose([1, 0, 2]))[None, :, :, None, :]
            # res = noises

            for _ in range(A.shape[-1]):
                sample = jnp.einsum("dij,dtbni->dtbnj", A, sample) + noises
                sample = (nodes_one_hot * values.transpose([1, 0, 2]))[
                    None, :, :, None, :
                ] + (1 - nodes_one_hot[None, :, :, None, :]) * sample
            res = sample
        else:
            # permute exogenous based on topsort
            noises = ((1 - nodes_one_hot[None, :, :, None, :]) * noises) + (
                nodes_one_hot * values.transpose([1, 0, 2])
            )[None, :, :, None, :]

            # noises = ((noises*(1-nodes_one_hot[..., None, None])) + (nodes_one_hot*values.transpose([1, 0])[..., None])[..., None, None])

            noises = jnp.einsum("dij,dtbsj->dtbsi", P, noises)
            # compute res using the upper triangular (toposorted graph) and the topsorted noises
            res = jnp.einsum(
                "dij,dtbni->dtbnj", jnp.linalg.inv(np.eye(N) - tri), noises
            )

            # permute res back to pre-toposort
            res = jnp.einsum("dij,dtbsj->dtbsi", P.transpose([0, 2, 1]), res)

        # THIS NOT NEEDED ANYMORE
        # res = res.transpose(1, 2, 4, 0, 3)
        # res = ((res*(1-nodes_one_hot[..., None, None])) + (nodes_one_hot*values.transpose([1, 0])[..., None])[..., None, None])
        # res = res.transpose([3, 0, 1, 4, 2])
        # dags x T x B x samples x D

        return res

    def batch_likelihood(
        self, nodes, datapoints, roll=False, onehot=False, weights=False
    ):
        # dags x designs x batch x samples x dims
        A = jnp.array([g.to_amat() for g in self.all_graphs])
        scales = np.array([g._variances**0.5 for g in self.all_graphs])
        DAGS, T, B, S, D = datapoints.shape

        # logpdfs = jnp.zeros((DAGS, DAGS, T, B, S))
        # outer x inner x designs x batch x samples
        # print("iterating over graphs")
        # for i, g in enumerate(self.all_graphs):
        #     print(i)
        #     locs = jnp.einsum('dtbsi,ii->dtbsi', datapoints, A[i])
        #     scale = self.all_graphs[i]._variances ** .5
        #     # permute scale based on topological ordering
        #     scale = scale[topological_sorts[i]]

        #     logpdfs = logpdfs.at[i].set(logpdf(datapoints, loc=locs, scale=scale).sum(-1))
        # print("done")

        if onehot:
            # transpose from B,T,N to T,B,N
            nodes_one_hot = nodes.transpose([1, 0, 2])
        else:
            nodes_one_hot = jnp.zeros((B * T, D))
            nodes_one_hot = nodes_one_hot.at[jnp.arange(B * T), nodes.ravel()].set(1)
            nodes_one_hot = nodes_one_hot.reshape(B, T, D).transpose([1, 0, 2])

        # ravel and wrap again
        locs = jnp.einsum("oij,etbsi->oetbsj", A, datapoints)

        #         (logpdf(_ds, loc=jnp.einsum('oij,etbsi->oetbsj', A, _ds), scale=scales[:, None, None, None, None, :])* (1-nodes_one_hot.transpose([1, 0, 2]))[
        # None, None, :, :, None, :] ).sum(-1)[0, 0, :, 0, 0]

        logpdfs = logpdf(
            datapoints, loc=locs, scale=scales[:, None, None, None, None, :]
        )

        # logpdf(datapoints[2], loc=(jnp.einsum('ij,tbsi->tbsj', A[2], datapoints[2])), scale=scales[2, None, None, :])[0, 13, :, 0]
        # import pdb; pdb.set_trace()
        # zero out logpdfs of the intervention node

        _mask = (1 - nodes_one_hot)[None, None, :, :, None, :]

        # import pdb; pdb.set_trace()
        # (logpdfs*_mask)[2, 2, 0, 13, 0, :].sum()

        # [47, 35, 42, 32, 22, 21, 19, 10, 20, 9, 6, 1, 36, 31, 41, 46, 39, 0, 18, 14, 34, 30, 43, 44, 23, 45, 48, 16, 13, 11, 37, 2, 17, 29, 12, 25, 7, 38, 8, 5, 3, 33, 27, 4, 24, 28, 49, 15, 40, 26]

        # and sum over dims
        logpdfs = (logpdfs * _mask).sum(-1)
        # _l = (logpdf(_ds, loc=jnp.einsum('oij,etbsi->oetbsj', A, _ds), scale=scales[:, None, None, None, None, :])*[_mask[0, 0]).sum(-1).transpose([3, 2, 1, 0, 4])[0]

        # 1 x 50 traj x 100 x 100 x 1 sample
        # B x Designs x Outer x Iner x Samples

        res = logpdfs.transpose([3, 2, 1, 0, 4])

        if weights:
            return res + self.log_weights[None, None, :, None, None]
        else:
            return res

    def sample(self):
        return self.amats[np.random.randint(len(self.amats))]

    def obs_entropy(self):
        pass

    def sample_interventions(self, nodes, value_samplers, nsamples):
        n_boot = len(self.dags)

        # Collect interventional samples
        # Bootstraps x Interventions x Samples x Nodes
        datapoints = np.array(
            [
                [
                    dag.sample_interventional({node: sampler}, nsamples=nsamples)
                    for node, sampler in zip(nodes, value_samplers)
                ]
                for dag in self.dags
            ]
        )

        return datapoints

    def interventional_likelihood(self, graph_ix, data, interventions):
        return self.dags[graph_ix].logpdf(data, interventions=interventions)

    def save(self, path):
        with open(os.path.join(path, "dags.pkl"), "wb") as b:
            pickle.dump(self.amats, b)
            b.close()

    def load(self, path):
        with open(os.path.join(path, "dags.pkl"), "rb") as b:
            self.amats = pickle.load(b)
            dags = [cd.DAG.from_amat(adj) for adj in self.amats]
            b.close()
        cov_mat = np.linalg.inv(self.precision_matrix)
        self.dags = [utils.cov2dag(cov_mat, dag) for dag in dags]
        self.all_graphs = self.dags
        self.functional_matrix = np.ones([len(self.dags), self.num_nodes])
