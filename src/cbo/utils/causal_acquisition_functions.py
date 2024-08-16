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




class CausalExpectedImprovement(Acquisition):
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

    # def evaluate(self, x: np.ndarray, method:str='UCB', Lambda=0) -> np.ndarray: # ! method
    def evaluate(self, x: np.ndarray, method:str='EI', Lambda=0) -> np.ndarray: # ! method
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

        if method == 'EI':
            u, pdf, cdf = get_standard_normal_pdf_cdf(self.current_global_min, mean, standard_deviation)
            if self.task == 'min':
                improvement = standard_deviation * (u * cdf + pdf)
            else:
                improvement = - (standard_deviation * (u * cdf + pdf))
        elif method == 'UCB':
            beta = Lambda * len(self.model.X[0]) * math.log(2*len(self.model.X)+1) # ! UCB 
            # print(len(variance)) # == 1
            delta = np.sqrt(beta * variance) if variance[0]>0 else 0
            # print(f'delta:{delta}') # ! DEBUG
            if self.task == 'min':
                improvement = mean + delta
            else:
                improvement = -mean + delta
        else:
            print('No such {} acquisition function implemented!'.format(method))
            exit()

        # print('beta:{}, delta:{}, mean:{}'.format(beta, delta, mean)) # ! DEBUG
        # print('X {} improvement is {}'.format(x, improvement)) # ! DEBUG
        return improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.

        :param x: locations where the evaluation with gradients is done.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)

        mean += self.jitter
        u, pdf, cdf = get_standard_normal_pdf_cdf(self.current_global_min, mean, standard_deviation)

        
        if self.task == 'min':
            improvement = standard_deviation * (u * cdf + pdf)
            dimprovement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx
        else:
            improvement = - (standard_deviation * (u * cdf + pdf))
            dimprovement_dx = -(dstandard_deviation_dx * pdf - cdf * dmean_dx)

        return improvement, dimprovement_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return isinstance(self.model, IDifferentiable)


def get_standard_normal_pdf_cdf(x: np.array, mean: np.array, standard_deviation: np.array) \
        -> Tuple[np.array, np.array, np.array]:
    """
    Returns pdf and cdf of standard normal evaluated at (x - mean)/sigma

    :param x: Non-standardized input
    :param mean: Mean to normalize x with
    :param standard_deviation: Standard deviation to normalize x with
    :return: (normalized version of x, pdf of standard normal, cdf of standard normal)
    """
    u = (x - mean) / standard_deviation
    pdf = scipy.stats.norm.pdf(u)
    cdf = scipy.stats.norm.cdf(u)
    return u, pdf, cdf
