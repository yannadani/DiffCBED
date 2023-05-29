import numpy as np
from collections import namedtuple


Data = namedtuple("data", ["samples", "nodes"])


class ReplayBuffer(object):
    def __init__(self, binary=False):
        self.buffer = []
        self.n_obs = 0
        self.n_int = 0
        self.binary = binary

    def update(self, data):
        self.buffer.append(data)

        if (self.binary and data.intervention_node.sum() == 0) or (not self.binary and data.intervention_node == -1):
            self.n_obs += len(data.samples)
        else:
            self.n_int += len(data.samples)

    def reset(self):
        self.buffer = []

    def data(self):
        interventions = []
        samples = []
        for item in self.buffer:
            interventions.extend([item.intervention_node] * item.samples.shape[0])
            samples.append(item.samples)

        if self.binary:
            return Data(samples=np.concatenate(samples), nodes=np.array(interventions).astype(bool))
        else:
            return Data(samples=np.concatenate(samples), nodes=np.array(interventions))
