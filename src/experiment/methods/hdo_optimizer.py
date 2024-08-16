from ...hdo import *
from ...sem.sem_synt import SEM_synt
from ...utils import *
import torch
Ackly10Problem=get_Ackly_problem(10)
Ackly2Problem=get_Ackly_problem(2)
Hartmann6Problem=get_Hartmann6_problem(6)
Hartmann10Problem=get_Hartmann6_problem(10)

def is_result_in_range(Xs:torch.tensor,bounds:torch.tensor):
    for col_index in range(Xs.shape[1]):
        col_values = Xs[:, col_index] 
        lower_bound = bounds[0, col_index]  
        upper_bound = bounds[1, col_index]
        if (not 
            (torch.all(col_values >= lower_bound) and torch.all(col_values <= upper_bound))
        ):
            return False
    return True

def get_hdo_from_hco(target_sem:SEM_synt,intervenable_variables=None):
    if(intervenable_variables is None):
        intervenable_variables=target_sem.get_intervenable_variables()
    def target_function(X):
        interv_plan=[]
        for i,x in enumerate(X):
            interv_plan.append((intervenable_variables[i],x))
        Y=target_sem.intervene(1,SEM_REPEATED_TIMES,interv_plan)
        return Y[0]
    return OptimizationProblem(
        function=target_function,
        bounds=target_sem.get_bounds(intervenable_variables),
        maximize=(target_sem.task=='max')
    )
import abc
import numpy as np


class HDO_Optimizer(abc.ABC):
    def __init__(self) -> None:
        self.init_point_budget=7#!Hyperparameter
        self.hdo_optimization_iter=300
    def reset_init_point_budget(self,init_point_budget):
        self.init_point_budget=init_point_budget
    def reset_hdo_optimization_iter(self,hdo_optimization_iter):
        self.hdo_optimization_iter=hdo_optimization_iter
    @abc.abstractmethod
    def __call__(self,problem:OptimizationProblem)->np.array:
        pass

class REMBO_Optimizer(HDO_Optimizer):#OK
    def __init__(self,dim_embedding):
        self.dim_embedding=dim_embedding
    def __call__(self,problem:OptimizationProblem):
        X,Ys=optimize_with_REMBO(
            problem=problem,
            dim_embedding=self.dim_embedding,
            init_point_budget=self.init_point_budget,
            optimize_budget=self.hdo_optimization_iter,
            )
        return X,Ys
class ALEBO_Optimizer(HDO_Optimizer):
    def __init__(self,dim_embedding):
        self.dim_embedding=dim_embedding
    def __call__(self,problem:OptimizationProblem):#Too Slow
        X,objects=optimize_with_ALEBO(
            problem=problem,
            dim_embedding=self.dim_embedding,
            init_point_budget=self.init_point_budget,
            optimize_budget=self.hdo_optimization_iter
        )
        return X,objects

class MCTSVS_Optimizer(HDO_Optimizer):
    def __init__(self,C_p):
        self.C_p=C_p
    def __call__(self,problem):#OK
        Xs,Ys=optimize_with_MCTSVS(#Special Attention
            problem=problem,
            Cp=self.C_p,
            optimize_budget=self.hdo_optimization_iter,
            )
        #assert(is_result_in_range(Xs,problem.bounds))
        return Xs,Ys
class TuRBO_Optimizer(HDO_Optimizer):
    def __init__(self):
        pass
    def __call__(self,problem):#OKOK
        X,fX=optimize_with_TuRBO(
            problem,#! Special Initialization
            optimize_budget=self.hdo_optimization_iter
        )
        return X,fX[0]
class CMAES_Optimizer(HDO_Optimizer):#! Maximize or minimize
    def __init__(self,sigma):
        self.sigma=sigma
    def __call__(self,problem):#OK
        X,fX=optimize_with_CMAES(
            problem,
            sigma=self.sigma,#! Special Initialization
            optimize_budget=self.hdo_optimization_iter
        )
        return X,fX
class DropoutBO_Optimizer(HDO_Optimizer):
    def __init__(self,k,is_add_ucb):
        self.k=k
        self.is_add_ucb=is_add_ucb
    def __call__(self,problem):#OK
        X,fX=optimize_with_DropoutBO(
            problem,
            self.k,
            init_point_budget=self.init_point_budget,
            optimize_budget=self.hdo_optimization_iter,
            is_add_ucb=self.is_add_ucb
        )
        return X,fX
class BO_Optimizer(HDO_Optimizer):
    def __init__(self,is_add_ucb) -> None:
        self.is_add_ucb=is_add_ucb
    def __call__(self,problem):
        X,fX=optimize_with_BO(problem,self.init_point_budget,optimize_budget=self.hdo_optimization_iter,is_add_ucb=self.is_add_ucb)
        return X,fX