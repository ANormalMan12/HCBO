from abc import ABC,abstractmethod
from joblib import Parallel, delayed
from ..utils import *
class SEM(ABC):
    def __init__(self):
        self.all_variable_space_torch=np.array([])
    def set_min_max_bounds(self,bounds):
        self.all_variable_space_torch=bounds
        print(self.all_variable_space_torch)
    def get_bounds(self,interv_set):
        min_val_to=self.all_variable_space_torch[0,interv_set].unsqueeze(0)
        max_val_to=self.all_variable_space_torch[1,interv_set].unsqueeze(0)
        ret_tensor= torch.cat([min_val_to,max_val_to],dim=0)
        return ret_tensor
    @abstractmethod
    def intervene(self,n_samples,repeated_times,interv:interv_plan):
        raise NotImplementedError("intervene function of SEM")
    #@abstractmethod
    #def get_data(self, intervention_set:interv_set):
    #    raise NotImplementedError("get_data function of SEM")
    #@abstractmethod
    #def get_all_data(self):
    #    raise NotImplementedError("get_all_data function of SEM")
    @abstractmethod
    def get_connection(self):
        raise NotImplementedError("get_connection function of SEM")
    @abstractmethod
    def get_intervenable_variables(self):
        raise NotImplementedError("get_intervenable_variables function of SEM")