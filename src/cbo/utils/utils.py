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
## Import emukit function
import emukit
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.acquisition import Acquisition
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.core.optimization import GradientAcquisitionOptimizer
import GPyOpt

import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from .cost_functions import *
from .causal_acquisition_functions import CausalExpectedImprovement
from .causal_optimizer import CausalGradientAcquisitionOptimizer


def get_new_dict_x(x_new, intervention_variables):
    x_new_dict = {}

    for i in range(len(intervention_variables)):
      x_new_dict[intervention_variables[i]] = x_new[0, i]
    return x_new_dict


def list_interventional_ranges(dict_ranges, intervention_variables):
    list_min_ranges = []
    list_max_ranges = []
    for j in range(len(intervention_variables)):
      list_min_ranges.append(dict_ranges[intervention_variables[j]][0])
      list_max_ranges.append(dict_ranges[intervention_variables[j]][1])
    return list_min_ranges, list_max_ranges


def get_interventional_dict(intervention_variables):
    interventional_dict = {}
    for i in range(len(intervention_variables)):
      interventional_dict[intervention_variables[i]] = ''
    return interventional_dict


def initialise_dicts(exploration_set, task):
    current_best_x = {}
    current_best_y = {}
    x_dict_mean = {}
    x_dict_var = {}
    dict_interventions = []


    for i in range(len(exploration_set)):
      variables = exploration_set[i]
      if len(variables) == 1:
        variables = variables[0]
      if len(variables) > 1:
        num_var = len(variables)
        string = ''
        for j in range(num_var):
          string += variables[j]
        variables = string

      ## This is creating a list of strings 
      dict_interventions.append(variables)


      current_best_x[variables] = []
      current_best_y[variables] = []

      x_dict_mean[variables] = {}
      x_dict_var[variables] = {}

      ## Assign initial values
      if task == 'min':
        current_best_y[variables].append(np.inf)
        current_best_x[variables].append(np.inf)
      else:
        current_best_y[variables].append(-np.inf)
        current_best_x[variables].append(-np.inf)
      
    return current_best_x, current_best_y, x_dict_mean, x_dict_var, dict_interventions


def add_data(original, new):
    data_x = np.append(original[0], new[0], axis=0)
    data_y = np.append(original[1], new[1], axis=0)
    return data_x, data_y


def find_current_global(current_y, dict_interventions, task):
    ## This function finds the optimal value and variable at every iteration
    dict_values = {}
    for j in range(len(dict_interventions)):
        dict_values[dict_interventions[j]] = []

    for variable, value in current_y.items():
        if len(value) > 0:
          if task == 'min':
            dict_values[variable] = np.min(current_y[variable])
          else:
            dict_values[variable] = np.max(current_y[variable])
    if task == 'min':        
      opt_variable = min(dict_values, key=dict_values.get)
    else:
      opt_variable = max(dict_values, key=dict_values.get)
    
    opt_value = dict_values[opt_variable]
    return opt_value

def interventional_simulation(list_interventional_ranges, manipulative_variables, intervention_num:int=50):
    simulation = np.random.rand(intervention_num, len(manipulative_variables))
    # print(f'Random Numpy: {simulation}') # ! DEBUG
    low_bound = np.array([list_interventional_ranges[v][0] for v in manipulative_variables])
    up_bound = np.array([list_interventional_ranges[v][-1] for v in manipulative_variables])
    return simulation * (up_bound - low_bound) + low_bound

def evaluate_GP_model(x:np.ndarray, model, task:str)->float:
    mean, variance = model.predict(x)
    beta = 10 * len(model.X[0]) * math.log(2*len(model.X)+1) # ! UCB
    delta = np.sqrt(beta * variance) if variance[0]>0 else 0
    if task == 'min':
        improvement = mean + delta
        return np.mean(np.sort(improvement)[:int(len(improvement)/10)+1])
    else:
        improvement = -mean + delta
        return np.mean(np.sort(improvement)[-int(len(improvement)/10)-1:])

def find_next_y_point_random(space, model):
    return np.array([space[0][i]+(space[1][i]-space[0][i])*random.random() for i in range(len(space[0]))])[np.newaxis,:]

def find_next_y_point_mine(space, model, current_global_best, evaluated_set, costs_functions, task = 'min'):
    ## This function optimises the acquisition function and return the next point together with the 
    ## corresponding y value for the acquisition function
    cost_acquisition = Cost(costs_functions, evaluated_set)
    # ! CBO
    optimizer = CausalGradientAcquisitionOptimizer(space)
    # acquisition = CausalExpectedImprovement(current_global_best, task, model)/cost_acquisition
    # acquisition = GPyOpt.acquisitions.EI_mcmc.AcquisitionEI(model, space) # ! 
    acquisition = CausalExpectedImprovement(current_global_best, task, model) # ! AcqFunc
    x_new, _ = optimizer.optimize(acquisition)
    y_acquisition = acquisition.evaluate(x_new)
    return y_acquisition, x_new
        
def find_next_y_point(space, model, current_global_best, evaluated_set, costs_functions, task = 'min', cost_avg=True):
    ## This function optimises the acquisition function and return the next point together with the 
    ## corresponding y value for the acquisition function
    cost_acquisition = Cost(costs_functions, evaluated_set)
    optimizer = CausalGradientAcquisitionOptimizer(space)
    acquisition = CausalExpectedImprovement(current_global_best, task, model)/cost_acquisition if cost_avg else CausalExpectedImprovement(current_global_best, task, model)
    #try:
    x_new, _ = optimizer.optimize(acquisition)
    y_acquisition = acquisition.evaluate(x_new)
    #except Exception as e:
    #  print(e)
    assert(y_acquisition == _)
    return y_acquisition, x_new

# parameter_list = [[1.,1.,0.0001, False], [1.,1.,10., False], [1.,1.,1., False], [1.,1.,1., False], [1.,1.,10., False], 
#                     [1.,1.,1., False], [1.,1.,10., False], [1.,1.,10., False], [1.,1.,10., False],[1.,1.,10., False]]
def fit_single_GP_model(X, Y, parameter_list, ard = False):
    kernel = RBF(X.shape[1], ARD = parameter_list[3], lengthscale=parameter_list[0], variance = parameter_list[1]) 
    gp = GPRegression(X = X, Y = Y, kernel = kernel, noise_var= parameter_list[2])
    gp.likelihood.variance.fix(1e-2)
    gp.optimize()
    return gp
def normallize_str(s):
    char_list = list(s)
    char_list.sort()
    result = ''.join(char_list)
    return result

def split_str(s):
   return [char for char in s]