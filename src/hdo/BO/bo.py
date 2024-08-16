from ..optimization_problem import *
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
#from botorch.models.transforms.input import Normalize
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
import numpy as np
import random
import gpytorch
def generate_random_vector(bounds:torch.tensor):
    vec=[]
    for i in range(bounds.shape[1]):
        vec.append(random.uniform(bounds[0][i],bounds[1][i]))
    return vec

def optimize_with_BO(
    problem:OptimizationProblem,
    init_point_budget,
    optimize_budget,
    is_add_ucb
    ):
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype=torch.get_default_dtype()
    def obj_function(X:torch.tensor):
        np_X:np.array=X.cpu().numpy()
        return torch.tensor(problem.function(np_X.flatten()),dtype=dtype).to(device)
    bounds=problem.bounds.to(dtype).to(device)
    data_x=torch.tensor([
        generate_random_vector(problem.bounds) for sample_time in range(init_point_budget)
    ],dtype=dtype).to(device)
    data_y=torch.tensor(
       [[obj_function(x)] for x in data_x]
    ,dtype=dtype).to(device)
    
    for i in range(optimize_budget):
        try:
            if(is_add_ucb):
                kernel=gpytorch.kernels.AdditiveStructureKernel(
                    base_kernel=gpytorch.kernels.RBFKernel(),
                    num_dims=bounds.shape[1]
                )
            else:
                kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            model = SingleTaskGP(
                covar_module=kernel,
                train_X=data_x,
                train_Y=data_y,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)  
        except Exception as e:
            print(e)
            print("Fail to fit GP")
            break
        new_x,acq_val=optimize_acqf(
            acq_function=UpperConfidenceBound(
                model,torch.tensor(0.1,dtype=dtype),maximize=problem.maximize
            ),
            bounds=bounds,
            q=1,
            num_restarts=15,
            raw_samples=512
        )
        #print(acq_val)
        new_y=obj_function(new_x)
        
        data_x=torch.cat([data_x,new_x], dim=0)
        data_y=torch.cat([data_y,torch.tensor([[new_y]],device=device)], dim=0)
    cpu_device=torch.device('cpu')
    
    X=[x.to(cpu_device) for x in  data_x[:].squeeze(0)]
    Y=[float(y.to(cpu_device)) for y in data_y[:].squeeze(0)]
    
    
    return X,Y