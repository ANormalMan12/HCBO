import botorch 
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
#from botorch.models.transforms.input import Normalize
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from typing import List
import random
import gpytorch
class DropoutBO():
    def __init__(
            self,
            fill_strategy:str,
            low_d:int,
            var_bounds,
            data_x,
            data_y,
            maximize:bool,
            is_add_ucb
            ):
        self.var_bounds=var_bounds
        assert(var_bounds.shape[0]==2)
        self.dim=var_bounds.shape[1]
        self.data_x=data_x
        self.data_y=data_y
        self.maximize=maximize
        self.low_d=low_d
        self.fill_strategy=fill_strategy
        self.is_add_ucb=is_add_ucb
    def add_data(self,x_new,y_new):
        self.data_x = torch.cat([self.data_x, x_new.unsqueeze(0)],dim=0)
        self.data_y = torch.cat([self.data_y, torch.tensor(y_new).unsqueeze(0).unsqueeze(1)],dim=0)

    def predict(self)->torch.Tensor:
        try_times=0
        yes=False
        while(not yes):
            try:
                selected_variables=random.sample([i for i in range(self.dim)], self.low_d)#Randomly selecet self.low_d dimesnions
                selected_data_x=self.data_x[:,selected_variables]
                selected_data_y=self.data_y[:]
                selected_var_bounds=self.var_bounds[:,selected_variables]
                if(self.is_add_ucb):
                    kernel=gpytorch.kernels.AdditiveStructureKernel(
                        base_kernel=gpytorch.kernels.RBFKernel(),
                        num_dims=self.dim
                    )
                else:
                    kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                model = SingleTaskGP(
                    selected_data_x,
                    selected_data_y,
                    covar_module=kernel
                )# Build Model on selected dimensions
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
                yes=True
            except Exception as e:
                print("Fail to fit, try to rerun:",try_times)
                try_times+=1
                if(try_times>=5):
                    raise e
                continue
        selected_x,acq_val=optimize_acqf(
            UpperConfidenceBound(
                model=model,
                beta=0.1,
                maximize=self.maximize
            ),
            bounds=selected_var_bounds,
            q=1,
            num_restarts=15,
            raw_samples=512,
        )
        selected_x=selected_x.squeeze()
        high_d=self.data_x.shape[1]
        ret_x=torch.zeros(high_d)
        if(self.maximize):
            best_index:int=self.data_y.argmax(dim=0)# Get the best y 's x
        else:
            best_index:int=self.data_y.argmin(dim=0)
        best_x=self.data_x[best_index].squeeze()
        if(self.fill_strategy=="copy"):
            for i in range(high_d):
                if i not in selected_variables:    
                    ret_x[i]=best_x[i]
                else:
                    if(self.low_d==1):
                        ret_x[i]=selected_x
                    else:
                        ret_x[i]=selected_x[selected_variables.index(i)]
        else:
            raise Exception("fill_strategy not supported")
        return ret_x