import numpy as np
from collections import defaultdict

from envs.samplers import Constant
from scipy.special import logsumexp
from .acquisition_strategy import AcquisitionStrategy


class FScoreBatchStrategy(AcquisitionStrategy):
    def _score_for_value(self, nodes, value_samplers):
        n_boot = len(self.model.dags)

        # DAGs x Interventions x Samples x Nodes - y[t][m]
        datapoints = self.model.sample_interventions(
            nodes, value_samplers, self.num_samples
        )

        mu_i_k = datapoints.mean(-2, keepdims=True)
        mu_k = mu_i_k.mean(0, keepdims=True)

        vbg_k = ((mu_i_k - mu_k) ** 2).sum((0, -1, -2))
        vwg_k = ((datapoints - mu_i_k) ** 2).sum((0, -1, -2))

        scores = vbg_k / vwg_k

        return scores, {}


class SoftFScoreStrategy(FScoreBatchStrategy):
    def __init__(self, model, env, args):
        super().__init__(model, env, args)
        self.temperature = args.bald_temperature

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