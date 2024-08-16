from ..utils import *
import gpytorch

class BOmodelAbstract(ABC):
    def __init__(self,variables:interv_set,var_bounds,data_x,data_y):
        if torch.cuda.is_available():
            device = torch.device("cuda") 
        else:
            device = torch.device("cpu")
        self.variables=variables
        self.var_bounds=var_bounds
        self.data_x=data_x.to(device)
        self.data_y=data_y.to(device)
        self.device=device
        self._update_model()
    def change_device(self,device):
        self.device=device
        self.data_x=self.data_x.clone().to(device)
        self.data_y=self.data_y.clone().to(device)
        self._update_model()

    @abstractmethod
    def add_data(self,x_new,y_new):
        pass
    
    @abstractmethod
    def _update_model(self):
        pass
    
    @abstractmethod
    def get_kernel(self):
        pass

class BOmodelRBF(BOmodelAbstract):
    def add_data(self,x_new,y_new):
        x_new=x_new.to(self.device)
        y_new=y_new.to(self.device)
        self.data_x = torch.cat([self.data_x, x_new],dim=0)
        self.data_y = torch.cat([self.data_y, y_new],dim=0)
        self._update_model()

    def _update_model(self):
        self.model = SingleTaskGP(
            covar_module=self.get_kernel(),
            train_X=self.data_x.to(self.device),
            train_Y=self.data_y.to(self.device),
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)  
    def get_kernel(self):
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

class BOmodelNormalizedRBF(BOmodelRBF):
    def _update_model(self):
        mean_y_val=self.data_y.mean()
        std_y_val=self.data_y.std()
        #std_y_val=1.0 if std_y_val<1e-6 else std_y_val
        new_y=(self.data_y-mean_y_val)/(std_y_val+1e-5)
        
        self.model = SingleTaskGP(
            covar_module=self.get_kernel(),
            train_X=self.data_x.to(self.device),
            train_Y=new_y.to(self.device)
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

class BOmodelAdditive(BOmodelRBF):
    def get_kernel(self):
        return gpytorch.kernels.AdditiveStructureKernel(
            base_kernel=gpytorch.kernels.RBFKernel(),
            num_dims=len(self.variables)
        )
class BOmodelNormalizedAdditive(BOmodelNormalizedRBF):
    def get_kernel(self):
        return gpytorch.kernels.AdditiveStructureKernel(
            base_kernel=gpytorch.kernels.RBFKernel(),
            num_dims=len(self.variables)
        )
    

def get_BO_model_class(is_linear:bool)->BOmodelAbstract:
    if(is_linear):
        return BOmodelAdditive
    else:
        return BOmodelRBF
def get_normalized_BO_model_class(is_linear:bool)->BOmodelAbstract:
    if(is_linear):
        return BOmodelNormalizedAdditive
    else:
        return BOmodelNormalizedRBF