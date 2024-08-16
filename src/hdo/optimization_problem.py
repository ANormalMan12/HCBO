
import torch
class OptimizationProblem():
    def __init__(self,function,bounds:torch.tensor,maximize:bool):
        assert(bounds.shape[0]==2)
        function(bounds[0])# test whether the function is legal
        function(bounds[1])
        self.function = function
        self.dim = bounds.shape[1]
        self.maximize=maximize
        self.bounds=torch.tensor(bounds)
        
    def get_lb_ub(self):
        return (self.bounds[0],self.bounds[1])
    
    def evaluate(self,X):
        return function(X)


