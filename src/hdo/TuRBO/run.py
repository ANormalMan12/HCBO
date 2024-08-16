from .turbo import Turbo1
import numpy as np
import torch
import math
import random
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
import time

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
def optimize_with_TuRBO(
    problem:OptimizationProblem,
    optimize_budget
):
    dim_high=problem.bounds.shape[1]
    lb=np.array(problem.bounds[0])
    ub=np.array(problem.bounds[1])
    maximize=problem.maximize
    def f(X):
        torch_X=torch.tensor(X)
        result=problem.function(torch_X)
        if(maximize):
            result=-result
        return result
    #start = time.time()#!Need changes
    turbo1 = Turbo1(
        f=f,  # Handle to objective function
        lb=lb,  # Numpy array specifying lower bounds
        ub=ub,  # Numpy array specifying upper bounds
        n_init=2*dim_high,  # Number of initial bounds from an Latin hypercube design
        max_evals=optimize_budget,  # Maximum number of evaluations
        batch_size=20,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=30,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cuda" if torch.cuda.is_available() else "cpu",  # "cpu" or "cuda"
        dtype="float32",  # float64 or float32
    )
    turbo1.optimize()
    #end = time.time()

    X=turbo1.X
    fX = turbo1.fX  # Observed values
    if(maximize):
        fX=-fX
    return X,np.array(fX).T