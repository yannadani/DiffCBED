import numpy as np
from scipy.special import logsumexp

from .acquisition_strategy import AcquisitionStrategy


class PCEBatchStrategy(AcquisitionStrategy):
    def _score_for_value(self, nodes, value_samplers):
        n_boot = len(self.model.dags)

        # DAGs x Interventions x Samples x Nodes - y[t][m]
        datapoints = self.model.sample_interventions(
            nodes, value_samplers, self.num_samples
        )

        info_gain = np.zeros(len(nodes))
        logpdfs = self.model._update_likelihood(
            nodes, self.num_samples, value_samplers, datapoints
        )
        logpdfs_np = logpdfs.values.swapaxes(0, 1)

        for outer_dag_ix, _ in enumerate(self.model.dags):
            numerator = logpdfs_np[outer_dag_ix, :, outer_dag_ix, :]

            idx = np.delete(np.arange(n_boot), outer_dag_ix)
            denom = logsumexp(logpdfs_np[outer_dag_ix][:, idx], axis=-2)
            info_gain += (numerator - denom).mean(-1)

        return info_gain, {}
