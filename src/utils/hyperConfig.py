## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns
from typing import *
import math 
import random
from abc import ABC, abstractmethod
## Import emukit function
import matplotlib.pyplot as plt
import emukit
import multiprocessing
#from emukit.core import ParameterSpace, ContinuousParameter
#from emukit.bayesian_optimization.acquisitions.expected_improvement import ExpectedImprovement
#from emukit.bayesian_optimization.acquisitions.probability_of_improvement import ProbabilityOfImprovement

#from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
#from emukit.core.optimization import GradientAcquisitionOptimizer
import GPyOpt
from abc import abstractmethod
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
#from botorch.models.transforms.input import Normalize
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
import os


MAX_TRY_TIME=30

def init_context(seed:int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_default_dtype(torch.float64)
    np.seterr(over='raise')

ACQ_RAW_SAMPLES=512
ACQ_NUM_RESTARTS=15
UCB_BETA=0.1

GLOBAL_PARALLEL_N_JOBS=1
SYNT_INIT_INTERVENTION_DATA_NUM=7
SEM_REPEATED_TIMES=2000
MAX_TIME_LIMIT=6*3600
