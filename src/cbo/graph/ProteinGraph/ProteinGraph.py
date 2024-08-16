## Import basic packages
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from collections import OrderedDict
from matplotlib import cm
import scipy
import itertools
import sys
from numpy.random import randn
import copy
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sklearn.mixture

from ..AbstractGraph import *
from emukit.core.acquisition import Acquisition

## Import GP python packages
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from .ProteinGraph_CostFunctions import define_costs

class ProteinGraph(GraphStructure):
    def __init__(self, true_observational_samples):

        true_C = np.asarray(true_observational_samples['C'])[:,np.newaxis]
        true_A = np.asarray(true_observational_samples['A'])[:,np.newaxis]
        true_R = np.asarray(true_observational_samples['R'])[:,np.newaxis]
        true_M = np.asarray(true_observational_samples['M'])[:,np.newaxis]
        true_Y = np.asarray(true_observational_samples['Y'])[:,np.newaxis]

        self.reg_A = LinearRegression().fit(true_C,                              true_A)
        self.reg_R   = LinearRegression().fit(np.hstack((true_C, true_A)),         true_R)
        self.reg_M   = LinearRegression().fit(np.hstack((true_C, true_A, true_R)), true_M)
        self.reg_Y   = LinearRegression().fit(np.hstack((true_A,true_M)),            true_Y)


        self.dist_C = np.random.normal(0,1)

        self.define_connection()

    def define_connection(self):
        self.name_num = {}
        self.num_name=['C','A','R','M','Y']
        self.name_num['C'] = 0
        self.name_num['A'] = 1
        self.name_num['R'] = 2
        self.name_num['M'] = 3
        self.name_num['Y'] = 4 # TOPO ORDER
        
        self.full_var=['C','A','R','M']

        self.connection = np.zeros((len(self.name_num), len(self.name_num)))
        for s in ['A','R','M']:
            self.connection[self.name_num['C']][self.name_num[s]] =1
        for s in ['R','M','Y']:
            self.connection[self.name_num['A']][self.name_num[s]] =1
        self.connection[self.name_num['R']][self.name_num['M']] = 1
        self.connection[self.name_num['M']][self.name_num['Y']] = 1
    def __str__():
        return "ProteinGraph"
    """
    def define_SEM(self):
        
        def fC(epsilon, **kwargs):
            return self.dist_C

        def fA(epsilon, C, **kwargs):
            X = np.ones((1,1))*C
            return np.float64(self.reg_A.predict(X))

        def fR(epsilon, C, A, **kwargs):
            X = np.ones((1,1))*np.hstack((C,A))
            return np.float64(self.reg_R.predict(X))
            #return value

        def fM(epsilon, C,A,R, **kwargs):
            X = np.ones((1,1))*np.hstack((C,A,R))
            return np.float64(self.reg_M.predict(X))
            #return value

        def fY(epsilon, A,M, **kwargs):
            X = np.ones((1,1))*np.hstack((A,M))
            return np.float64(self.reg_Y.predict(X))
            #return value

        graph = OrderedDict ([
          ('C', fC),
          ('A', fA),
          ('R', fR),
          ('M', fM),
          ('Y', fY)
        ])
        return graph
    """
    def get_interventional_ranges(self):
        min_intervention_C = -10  #matter data
        max_intervention_C = 10

        min_intervention_A  = -1000 #matter data
        max_intervention_A = 2000

        min_intervention_M = -5  # matter data
        max_intervention_M = 2

        
        dict_ranges = OrderedDict ([
          ('C', [min_intervention_C, max_intervention_C]),
          ('A', [min_intervention_A, max_intervention_A]),
#          ('R', [min_intervention_R, max_intervention_R]),
          ('M', [min_intervention_M, max_intervention_M]),
        ])
        return dict_ranges

    def get_sets(self):
        return {"manu_var": ['C', 'A', 'M']}
    
    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs