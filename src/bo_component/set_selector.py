from ..utils import *
from .BOmodel import *
from .acqusition import *

class SetSelectorAbstract(ABC):
    @abstractmethod
    def predict_set(self,**kwargs)->int:
        pass
    @abstractmethod
    def get_info(self)->dict:
        pass
class SetSelector_Random(SetSelectorAbstract):
    def __init__(self,lenES:int):
        self.lenES=lenES
    def predict_set(self,**kwargs)->int:
        return random.randint(0,self.lenES-1)
    def get_info(self)->dict:
        return {"type":"random"}


def get_possible_alpha_array(mean_list,acq_list):
    mean_array=np.array(mean_list)
    mean_min=np.min(mean_array)
    mean_max=np.max(mean_array)
    #acq_array=np.array(acq_list)
    #acq_min=np.min(acq_array)
    possible_alpha_list=[(mean_max-mean_min)/(acq+1e-7) for acq in acq_list]
    return np.array(possible_alpha_list)

def get_alpha_average(possible_alpha_array):
    return np.mean(possible_alpha_array)
def get_alpha_median(possible_alpha_array):
    return np.median(possible_alpha_array)

class ISSFstrategyAbstract(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def __str__(self)->str:
        raise("__str__ not implemented")
    @abstractmethod
    def get_issf_list(self,mean_list,acq_list):
        raise("get_issf_list not implemented")
    @abstractmethod
    def need_mean_list(self)->bool:
        raise("need_mean_list not implemented")
    @abstractmethod
    def need_acq_list(self)->bool:
        raise("need_acq_list not implemented")
class ISSFstrategyAlpha(ISSFstrategyAbstract):
    def __init__(self,get_alpha_function,times_to_update_alpha:int):
        self.visit_times=times_to_update_alpha
        self.get_alpha_function=get_alpha_function
        self.times_to_update_alpha=times_to_update_alpha
    def __str__(self):
        return "Median-"+str(self.times_to_update_alpha)
    def get_issf_list(self,mean_list,acq_list,really_update=True):
        if(self.visit_times>=self.times_to_update_alpha):
            possible_alpha_array=get_possible_alpha_array(mean_list=mean_list,acq_list=acq_list)
            self.alpha=self.get_alpha_function(possible_alpha_array)
            self.visit_times=0
        if(really_update):
            self.visit_times+=1
        issf_list=[mean+self.alpha*acq for mean,acq in zip(mean_list,acq_list)]
        return issf_list
    def need_acq_list(self) -> bool:
        return True
    def need_mean_list(self) -> bool:
        return True
class ISSFstrategyAlphaFixed(ISSFstrategyAbstract):
    def __init__(self,fixed_alpha):
        self.alpha=fixed_alpha
    def __str__(self):
        return "FixedAlpha-"+str(self.alpha)
    def get_issf_list(self,mean_list,acq_list,really_update=True):
        issf_list=[mean+self.alpha*acq for mean,acq in zip(mean_list,acq_list)]
        return issf_list
    def need_acq_list(self) -> bool:
        return True
    def need_mean_list(self) -> bool:
        return True

class ISSFstrategyAcquisition(ISSFstrategyAbstract):
    def __init__(self):
        pass
    def __str__(self):
        return "Acquisition"
    def get_issf_list(self,mean_list,acq_list,really_update=True):
        return copy.deepcopy(acq_list)
    def need_acq_list(self) -> bool:
        return True
    def need_mean_list(self) -> bool:
        return False
class ISSFstrategyMeanY(ISSFstrategyAbstract):
    def __init__(self):
        pass
    def __str__(self):
        return "MeanY"
    def get_issf_list(self,mean_list,acq_list,really_update=True):
        return copy.deepcopy(mean_list)
    def need_acq_list(self) -> bool:
        return False
    def need_mean_list(self) -> bool:
        return True



class SetSelector_BO(SetSelectorAbstract):
    def __init__(self,lenES,acq_func_type,issf_strategy:ISSFstrategyAbstract):
        """
        acq_func_type: EI, PI, UCB
        """
        self.lenES=lenES
        self.acq_history=[]
        self.mean_history=[]
        self.issf_history=[]
        self.acq_func_type=acq_func_type
        self.issf_strategy=issf_strategy
        if(self.acq_func_type=="EI"):
            self.acqusition_function=find_EI_acq_val
        elif(self.acq_func_type=="PI"):
            self.acqusition_function=find_PI_acq_val
        elif(self.acq_func_type=="UCB" or self.acq_func_type=="UCBn"):
            self.acqusition_function=find_GP_UCB_acq_val
        else:
            raise ValueError("func_type not supported")
    
    def __str__(self):
        return self.acq_func_type+'-'+str(self.issf_strategy)
    def disable_index(self,index):
        if(not len(self.acq_history)==0):
            self.acq_history[-1][index]=-np.inf
        if(not len(self.mean_history)==0):
            self.mean_history[-1][index]=-np.inf
        if(not len(self.issf_history)==0):
            self.issf_history[-1][index]=-np.inf
        
    def get_next_acq_list(self,BOmodelList:List[BOmodelAbstract],maximize,last_update_index,global_best_y):
        """If global_best_y is None, then use local best y as best value"""
        if(len(self.acq_history)==0):
            acq_list=[]
            for i,BOmodel_item in enumerate(BOmodelList):
                acq_list.append(
                    get_x_acq_item(
                        self.acqusition_function,BOmodel_item,maximize,global_best_y
                    )[1].item()
                )
        else:
            acq_list=copy.deepcopy(self.acq_history[-1])
            acq_list[last_update_index]=get_x_acq_item(
                self.acqusition_function,BOmodelList[last_update_index],maximize,global_best_y
            )[1].item()
        acq_array=np.array(acq_list)
        min_acq_val=np.min(acq_array)
        if(min_acq_val<0):
            acq_list=[acq-min_acq_val+(1e-7) for acq in acq_list]
        return acq_list
    def get_next_mean_list(self,BOmodelList:List[BOmodelAbstract],last_update_index):
        if(len(self.mean_history)==0):
            mean_list=[]
            for i,BOmodel_item in enumerate(BOmodelList):
                mean_list.append(
                    BOmodel_item.data_y.mean().item()
                )
        else:
            mean_list=copy.deepcopy(self.mean_history[-1])
            mean_list[last_update_index]=BOmodelList[last_update_index].data_y.mean().item()
        return mean_list
    
    def predict_set(self,BOmodelList:List[BOmodelAbstract],maximize,last_index,global_best_y,**kwargs)->int:
        if(self.issf_strategy.need_acq_list()):
            acq_list=self.get_next_acq_list(BOmodelList,maximize=maximize,last_update_index=last_index,global_best_y=global_best_y)
            self.acq_history.append(acq_list)
        else:
            acq_list=None
        if(self.issf_strategy.need_mean_list()):
            mean_list=self.get_next_mean_list(BOmodelList,last_index)
            self.mean_history.append(mean_list)
        else:
            mean_list=None        
        issf_list=self.issf_strategy.get_issf_list(mean_list,acq_list)
        self.issf_history.append(issf_list)
        
        return np.argmax(issf_list)# For botorch, max acqf value is always the best one.

    def get_info(self)->dict:
        return {
            "Set_Selector_Type":self.__str__(),
            "acq_history":self.acq_history,
            "mean_history":self.mean_history,
            "issf_history":self.issf_history
        }