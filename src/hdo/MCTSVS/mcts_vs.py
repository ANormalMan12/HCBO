# The original code is from https://github.com/lamda-bbo/MCTS-VS

import torch
import botorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
import datetime
import time
import os
# from benchmark import get_problem
from .MCTSVS.MCTS import MCTS
# from utils import save_args
from ..optimization_problem import OptimizationProblem
def optimize_with_MCTSVS(
    problem:OptimizationProblem,
    Cp,
    optimize_budget:int):

    def f(X):
        X_torch=torch.tensor(X)
        return problem.function(X_torch)
    dim_high=problem.dim
    lb=np.array(problem.bounds[0])
    ub=np.array(problem.bounds[1])
    maximize=problem.maximize

    feature_batch_size = 2# N_v
    sample_batch_size = 3# N_s
    min_num_variables = 3# N_split
    select_right_threshold = 5#N_bad
    turbo_max_evals = 20 #MCTS-VS-TuRBO
    k = 20
    
    ipt_solver = 'bo'
    uipt_solver = 'bestk'

    agent = MCTS(
        func=f,
        dims=dim_high,
        lb=lb,
        ub=ub,
        feature_batch_size=feature_batch_size,
        sample_batch_size=sample_batch_size,
        Cp=Cp,
        min_num_variables=min_num_variables,
        select_right_threshold=select_right_threshold,
        k=k,
        split_type='mean',
        ipt_solver=ipt_solver,
        uipt_solver=uipt_solver,
        turbo_max_evals=turbo_max_evals,
        maximize=maximize
    )

    agent.search(max_samples=optimize_budget+sample_batch_size*feature_batch_size, verbose=False)
    print("MCTS-VS best sample",agent.curt_best_sample)
    x_array=[]
    y_array=[]
    for i in range(len(agent.samples)):
        x_array.append(agent.samples[i][0])
        y_array.append(agent.samples[i][1])
    return np.array(x_array),np.array(y_array)
    