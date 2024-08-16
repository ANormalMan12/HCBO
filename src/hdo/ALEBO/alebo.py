import numpy as np
import torch
import time
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.service.managed_loop import optimize
from ..optimization_problem import OptimizationProblem


import math
def optimize_with_ALEBO(
    problem:OptimizationProblem,
    dim_embedding:int,
    init_point_budget,
    optimize_budget,
    ):
    #device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device=torch.device("cpu") # CUDA is with bugs
    dim_low=dim_embedding
    dim_high=problem.dim
    maximize=problem.maximize
    target=problem.function

    parameters = [
        {"name": f"x{i}",
        "type": "range",
        "bounds": [float(problem.bounds[0][i]),float(problem.bounds[1][i])],
        "value_type": "float"
        }
        for i in range(dim_high)
        ]

    def obj_func(parameterization):
        X = np.array([parameterization[f'x{i}'] for i in range(dim_high)])
        Y = float(target(X))
        print(Y)
        return {"obj":(Y, 0.0)}

    alebo_strategy = ALEBOStrategy(D=dim_high, d=dim_low, init_size=init_point_budget,device=device)
    start = time.time()

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=obj_func,
        objective_name="obj",
        minimize=not maximize,
        total_trials=optimize_budget,
        generation_strategy=alebo_strategy,
    )
    X_list=[]
    Y_list=[]
    for i,trial in experiment.trials.items():
        X_dict=trial.arm.parameters
        x=[None]*len(X_dict)
        for key,value in X_dict.items():
            x[int(key[1:])]=value
        for val in x:
            assert(x is not None)
        X_list.append(x)
        Y_list.append(trial.objective_mean)
    print("best_parameters,",best_parameters)
    print(type(best_parameters))
    assert(len(X_list)==len(Y_list))
    return X_list,Y_list