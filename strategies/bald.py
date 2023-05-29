import os
from collections import defaultdict

import jax.numpy as jnp
import numpy as np
import tqdm
from scipy.special import logsumexp

from envs.samplers import Constant

from .acquisition_strategy import AcquisitionStrategy


def entropy(p):
    return -sum(p * np.log(p))


def logmeanexp(A, axis):
    return logsumexp(A, axis=axis) - np.log(A.shape[axis])


# bring random acq on this
# Random, BALD, ABCD, Biology, F-score
class BALDStrategy(AcquisitionStrategy):
    def _score_for_value(self, nodes, value_samplers, datapoints=None):
        # DAGs x Interventions x Samples x Nodes - y[t][m]
        n_samples = self.num_samples
        if datapoints is None:
            datapoints = self.model.sample_interventions(
                nodes, value_samplers, n_samples
            )

        logpdfs = self.model._update_likelihood(
            nodes, n_samples, value_samplers, datapoints
        )

        # Nodes x Inner x Outer x Samples
        logpdfs_val = logpdfs.values

        n_boot = logpdfs_val.shape[1]
        # num = np.zeros([len(nodes), n_boot, n_samples])
        # denom = np.zeros([len(nodes), n_boot, n_samples])
        # for intv_ix in range(len(nodes)):
        #     for outer_dag_ix in range(n_boot):
        #         num[intv_ix, outer_dag_ix] = logpdfs.sel(
        #             outer_dag=outer_dag_ix, inner_dag=outer_dag_ix, intervention_ix=intv_ix
        #         )

        #         tmp = logpdfs.sel(
        #                 outer_dag=outer_dag_ix, intervention_ix=intv_ix
        #             )
        #         denom[intv_ix, outer_dag_ix] = logmeanexp(tmp, 0)

        # MI = (num-denom).mean((1, 2))

        MI = (
            np.diagonal(logpdfs_val[:, None], axis1=2, axis2=3).sum(1).swapaxes(1, 2)
            - logmeanexp(logpdfs_val[:, None].sum(1), 2)
        ).mean((1, 2))

        return MI, {"logpdfs": logpdfs_val}


class DiffBatchBALDStrategy(AcquisitionStrategy):
    def _score(self, designs):
        # return the I(y_1, ..., d_n ; \theta | d_1, ..., d_n)
        return MI

    def acquire(self, nodes):
        designs = self.init_designs()

        for _ in range(epochs):
            # optimizer designs to minimize -eig
            eig = self._score(designs)
            loss = (-eig).mean()

        return designs


class BatchBALDStrategy(BALDStrategy):
    def _score_for_value(self, nodes, value_samplers, pnm1):
        # DAGs x Interventions x Samples x Nodes - y[t][m]
        n_samples = self.num_samples
        datapoints = self.model.sample_interventions(nodes, value_samplers, n_samples)

        logpdfs = self.model._update_likelihood(
            nodes, n_samples, value_samplers, datapoints
        )

        # intervention x inner dag x outer dag x samples
        logpdfs_val = logpdfs.values

        if pnm1 is not None:
            # the recursive equation (13) in batchbald paper
            logpdfs_val += pnm1[None, ...]

        total_uncertainty = -logmeanexp(logpdfs_val, axis=2).mean((1, 2))
        aleatoric_uncertainty = -np.diagonal(logpdfs_val, axis1=1, axis2=2).mean((1, 2))

        MI = total_uncertainty - aleatoric_uncertainty

        return MI, {"logpdfs": logpdfs_val}

    def acquire(self, nodes, iteration):
        n_boot = len(self.model.dags)
        current_logpdfs = np.zeros([n_boot, n_boot])

        # DAGs x Interventions x Samples x Nodes - y[t][m]
        strategy = self.get_value_strategy(nodes)

        targets = []
        values = []
        pnm1 = None
        scores = []
        for _ in range(self.args.batch_size):
            strategy(
                self._score_for_value,
                n_iters=self.args.num_intervention_values,
                pnm1=pnm1,
            )

            pnm1 = strategy.extra[strategy.max_iter]["logpdfs"][strategy.max_j]

            targets.append(strategy.max_j)
            values.append(strategy.max_x)
            scores.append(strategy.target)

        selected_interventions = defaultdict(list)
        for value, target in zip(values, targets):
            selected_interventions[target].append(Constant(value))

        return selected_interventions, scores


class SoftBALDStrategy(BALDStrategy):
    def __init__(self, model, env, args):
        super().__init__(model, env, args)
        self.temperature = args.temperature

    def acquire(self, nodes, iteration):
        # DAGs x Interventions x Samples x Nodes - y[t][m]
        strategy = self.get_value_strategy(nodes)
        strategy(self._score_for_value, n_iters=self.args.num_intervention_values)

        probs = (
            np.exp(strategy.target / self.temperature)
            / np.exp(strategy.target / self.temperature).sum()
        )

        assert (
            self.batch_size <= self.args.num_nodes
        ), "Batch size need to be smaller than the number of nodes"
        try:
            interventions = np.random.choice(
                range(len(probs.flatten())),
                p=probs.flatten(),
                replace=False,
                size=self.batch_size,
            )
            value_ids, node_ids = np.unravel_index(interventions, shape=probs.shape)
        except ValueError:
            value_ids, node_ids = np.unravel_index(
                np.nonzero(probs.flatten())[0], shape=probs.shape
            )
            interventions = np.random.choice(
                range(len(probs.flatten())),
                p=probs.flatten(),
                replace=True,
                size=self.batch_size - len(value_ids),
            )
            value_ids_, node_ids_ = np.unravel_index(interventions, shape=probs.shape)

            value_ids = np.concatenate([value_ids, value_ids_])
            node_ids = np.concatenate([node_ids, node_ids_])

        selected_interventions = defaultdict(list)
        for value_id, node in zip(value_ids, node_ids):
            selected_interventions[nodes[node]].append(
                Constant(strategy.values[value_id][nodes[node]])
            )

        return selected_interventions, probs
