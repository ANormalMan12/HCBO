"""
import numpy as np
class Distribution:
    def __init__(self):
        pass
    def sample(self, n_sample):
        raise NotImplementedError("sample method not implemented")

class NormalDistribution(Distribution):
    def __init__(self, mean=0, std_dev=1):
        self.mean = mean
        self.std_dev = std_dev

    def sample(self, n_sample):
        return np.random.normal(self.mean, self.std_dev,n_sample)
"""