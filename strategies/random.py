import os
import numpy as np

from envs.samplers import Constant
from .acquisition_strategy import AcquisitionStrategy
from collections import defaultdict


class RandomAcquisitionStrategy(AcquisitionStrategy):
    def acquire(self, nodes, iteration):
        strategy = self.get_value_strategy(nodes)
        values = strategy.intervention_value_prior(self.args.batch_size)
        values = values.reshape(self.args.batch_size, -1)

        if self.args.num_targets > 0:
            nodes = np.zeros((self.args.batch_size, self.args.  num_nodes))

            for i in range(self.args.batch_size):
                nodes[i][np.random.choice(np.arange(nodes.shape[-1]), self.args.num_targets, replace=False)] = True
        else:
            nodes = np.random.choice([True, False], size=self.args.batch_size*self.args.num_nodes).reshape(self.args.batch_size, self.args.num_nodes)

        return {'nodes': nodes, 'values': values}, None