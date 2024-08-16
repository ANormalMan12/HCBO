## Import basic packages
import numpy as np
import math
import scipy.stats
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns

from typing import Tuple, Union
from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.acquisition import Acquisition

import emukit
from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.optimization.context_manager import ContextManager


from emukit.core.optimization.anchor_points_generator import AnchorPointsGenerator




class UCBMC(Acquisition):
    def __init__(self, current_global_min, task, model: Union[IModel, IDifferentiable], jitter: float = float(0))-> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """
        self.model = model
        self.jitter = jitter
        self.current_global_min = current_global_min
        self.task = task

    # def evaluate(self, x: np.ndarray, method:str='UCB') -> np.ndarray: # ! method
    def evaluate(self, x: np.ndarray, Lambda=0) -> np.ndarray: # ! method
        #print('##### CausalExpectedImprovement')
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """

        mean, variance = self.model.predict(x)  # (sample_num, dim)
        if variance[0,0] < 1e-10:  #! 
            standard_deviation = np.full((1,1), 1e-10)
        else:
            standard_deviation = np.sqrt(variance)
        mean += self.jitter

        beta = Lambda * len(self.model.X[0]) * math.log(2*len(self.model.X)+1) # ! UCB 
        # print(len(variance)) # == 1
        delta = np.sqrt(beta * variance) if variance[0]>0 else 0
        if self.task == 'min':
            improvement = mean + delta
        else:
            improvement = -mean + delta

        return improvement
