import abc
## Import basic packages
import numpy as np
import pandas as pd
from matplotlib import pylab as plt

from matplotlib import cm
import scipy
import itertools
import sys
from numpy.random import randn
import copy
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sklearn.mixture
from emukit.core.acquisition import Acquisition
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from collections import OrderedDict, deque

class GraphStructure:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def define_SEM():
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def get_interventional_ranges(self):
        raise NotImplementedError("Subclass should implement this.")
    
    @abc.abstractmethod
    def get_sets(self):
        raise NotImplementedError("Subclass should implement this.")

