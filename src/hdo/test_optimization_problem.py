import botorch
import torch
from .optimization_problem import OptimizationProblem


def Hartmann_6_function(X):
    X=torch.tensor(X)
    Hartmann_function=botorch.test_functions.Hartmann(6)
    return Hartmann_function.evaluate_true(X[:6])
def get_Hartmann6_problem(dim:int):
    assert(dim>=6)
    bounds=torch.tensor([
        [0.0 for i in range(dim)],
        [1.0 for i in range(dim)]   
    ])
    return OptimizationProblem(Hartmann_6_function,bounds,maximize=False)

def Ackly_2_function(X):
    Ackly_function=botorch.test_functions.Ackley(2)
    return Ackly_function.evaluate_true(X[:2])

def get_Ackly_problem(dim:int):
    assert(dim>=2)
    bounds=torch.tensor([
        [-32.768 for i in range(dim)],
        [32.768 for i in range(dim)]   
    ])
    return OptimizationProblem(Ackly_2_function,bounds,maximize=False)