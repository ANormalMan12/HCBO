import os
import datetime
import time

from ..core import ObjFunc
from ..core.MCTS import MCTS
from ..core.BayeOpt import init_points_dataset_bo, next_point_bo, update_dataset_ucb
from ..core.RandEmbed import generate_random_matrix, random_embedding
import torch
import gpytorch
import numpy as np
import random


torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)
def no_growth(x, num):
    x_no = torch.ones(num) * -1e5
    # print(x)
    for i in range(x[0].shape[0]):
        x_no[i] = x[0, i]
    # x_no = x[0].clone()
    for i in range(1, x_no.shape[0]):
        x_no[i] = x_no[i-1:i+1].max()
    # print(x_no)
    return x_no

from ..optimization_problem import OptimizationProblem
    # bounds = torch.tensor([[-32.768, 32.768]] * 2)
def optimize_with_REMBO(
    problem:OptimizationProblem,
    dim_embedding:int,
    init_point_budget:int,
    optimize_budget:int
    ):
    dim_low=dim_embedding
    dim_high=problem.bounds.shape[1]
    bounds_high=np.array(problem.bounds.t())
    maximize=problem.maximize
    kernel_type = 'matern'
    sigma = [1.0] * dim_high
    bounds_low = torch.tensor([[-torch.sqrt(torch.tensor(dim_low)), torch.sqrt(torch.tensor(dim_low))]] * dim_low)
    def REMBO_Obj_Function(X):
        retL=[]
        for x in X:
            retL.append(problem.function(x))
        return torch.tensor(retL)


    rand_mat = generate_random_matrix(dim_low, dim_high, sigma)
    dataset = init_points_dataset_bo(init_point_budget, rand_mat, bounds_low, bounds_high, REMBO_Obj_Function)

    #result_x_list=[]
    #result_target_list=[]
    for i in range(optimize_budget):
        beta = 0.2 * dim_low * torch.log(torch.tensor(2 * (i + 1)))
        flag, next_y = next_point_bo(dataset, beta, bounds_low, kernel_type,maximize)
        if flag:
            next_x = random_embedding(next_y, rand_mat, bounds_high)
            next_f = REMBO_Obj_Function(next_x)
            dataset = update_dataset_ucb(next_y, next_x, next_f, dataset)
            print(f'Iteration: {i}', next_f)
    #        result_x_list.append(next_x.squeeze())
    #        result_target_list.append(next_f.squeeze())
        else:
            break

    return dataset['x'].tolist(),dataset['f'].squeeze(1).tolist()
