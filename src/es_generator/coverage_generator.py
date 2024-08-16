from ..sem.sem import *
from abc import abstractmethod
from typing import *
from ..utils import *
from ..bo_component import *
from itertools import combinations
from collections import deque
from joblib import Parallel, delayed

def generate_sorted_K_combinations(input:interv_set,K):
    characters = list(input)
    combinations_list = list(combinations(characters, K))
    combinations_list= sorted(combinations_list)
    return tuple(combinations_list)
def calc_inDegree(connection):
    num_variables=connection.shape[0]
    inD=np.zeros(num_variables,dtype=int)
    for i in range(num_variables):
        for j in range(num_variables):
            if(connection[i,j]):
                inD[j]+=1
    return inD
class ESmodule():
    def __init__(self,theSEM:SEM) -> None:
        self.weight_dict={}
        self.coverage_type_dict={}
    def set_covreage_type_dict(self,coverage_type_dict):
        self.coverage_type_dict=coverage_type_dict
    def set_weight_dict(self,weight_dict):
        self.weight_dict=weight_dict
    def set_x_y_history(self,x_y_history):
        self.x_y_history=x_y_history
def get_bestES_per_dimension(manu_var_list,coverage_dict):
    manu_var_list=sorted(list(manu_var_list))
    toDump={}
    for l in range(1,len(manu_var_list)+1):
        max_kv = max(coverage_dict.items(),
                    key=lambda item: item[1] if len(item[0])==l else float('-inf'))
        toDump[max_kv[0]]=max_kv[1]
    return [tuple(sorted(list(key))) for key in  toDump.keys()]

def get_weight_and_data_x_y_var_dict(theSEM:SEM,weight_explore_times:int,weight_bo_init_sample:int):
    Ystar_dict={}
    
    data_x_y_var_dict={}
    for singleVar in theSEM.get_intervenable_variables():#forall except Y
        print("Dealing with",singleVar)
        maximize=(theSEM.task=='max')
        if(singleVar in Ystar_dict):
            raise ValueError("Duplicate singleVar")
        data_x=torch.tensor([])
        data_y=torch.tensor([])
        for i in range(weight_explore_times):
            if(i<weight_bo_init_sample):
                new_x=[float(random.uniform(
                    theSEM.get_bounds((singleVar,))[0],
                    theSEM.get_bounds((singleVar,))[1]
                ))]
            else:
                try:
                    single_model=BOmodelRBF(variables=(singleVar,),
                        var_bounds=torch.tensor(theSEM.get_bounds((singleVar,))),
                        data_x=data_x,
                        data_y=data_y
                    )
                except Exception as e:
                    print("Fail in",singleVar)
                    print(e)
                    break 
                #if(maximize):
                #    best_f=data_y.max()
                #else:
                #    best_f=data_y.min()
                new_x,___=find_GP_UCB_acq_val(single_model.model,single_model.var_bounds,
                            maximize)
                new_x=[float(new_x[0])]
                for row in data_x:
                    if(torch.allclose(torch.tensor(new_x),row)):
                        break
            new_y=theSEM.intervene(1,SEM_REPEATED_TIMES,[(singleVar,new_x[0])])
            if(data_x.shape[0]==0):
                data_x=torch.tensor([new_x])
                data_y=torch.tensor([new_y])
            else:
                data_x=torch.cat([data_x,torch.tensor([new_x])], dim=0)
                data_y=torch.cat([data_y,torch.tensor([new_y])], dim=0)
        if(theSEM.task=='max'):
            Ystar_dict[singleVar]=torch.max(data_y).item()
            #for max task, use max value
        elif(theSEM.task=='min'):
            Ystar_dict[singleVar]=-torch.min(data_y).item()
            #for min task, use min value 's negative
        else:
            raise BaseException("Illegal task")
        data_x_y_var_dict[singleVar]=[data_x,data_y]
    weight_dict=get_weight_dict(Ystar_dict)
    return weight_dict,data_x_y_var_dict

def get_weight_dict(Ystar_dict):
    #normalize to maximum function value
    global_min=min(Ystar_dict.values())
    global_max=max(Ystar_dict.values())
    epsilon=abs(global_max-global_min)*0.01
    weight_dict={}
    weight_dict = {num:(val-global_min+epsilon) for num, val in Ystar_dict.items()}
    normalize_sum = sum(list(weight_dict.values()))
    weight_dict = {num:val/normalize_sum for num, val in weight_dict.items()}
    weight_dict = {num:val*1e7 for num, val in weight_dict.items()}
    return weight_dict

def add_new_variable_to_set(intervention_set:interv_set,new_var:int):
    retL=list(intervention_set)
    retL.append(new_var)
    return tuple(sorted(retL))

def get_K_greed(manu_var,cal_a_set,coverage_dict,K:int=3):
    assert(K>=3)
    for var in manu_var:
        s=(var,)
        coverage_dict[s]=cal_a_set(s)
        for ano_var in manu_var:
            if(var==ano_var):
                continue
            two_s=add_new_variable_to_set(s,ano_var)
            if(two_s not in coverage_dict):
                coverage_dict[two_s]=cal_a_set(two_s)
    def get_all_near(intervention_set):
        possible_next_set_list=[]
        for var in manu_var:
            if(var in intervention_set):
                continue
            possible_next_set=add_new_variable_to_set(intervention_set,var)
            possible_next_set_list.append(possible_next_set)
        return possible_next_set_list
    def get_best_next(possible_next_set_list):
        best_next_set=None
        best_next_set_coverage=-np.inf
        for s in possible_next_set_list:
            if(best_next_set_coverage<coverage_dict[s]):
                best_next_set=s
                best_next_set_coverage=coverage_dict[s]
        return best_next_set
    
    initial_exploration_set=generate_sorted_K_combinations(manu_var,K)
    
    coverage_initial_results=Parallel(n_jobs=GLOBAL_PARALLEL_N_JOBS)(
        delayed(cal_a_set)(intervention_set) for intervention_set in initial_exploration_set)
    for i,intervention_set in enumerate(initial_exploration_set):
        coverage_dict[intervention_set]=coverage_initial_results[i]
    exploration_set_list=initial_exploration_set

    while(True):
        exploration_set_list=list(set(exploration_set_list))
        print(exploration_set_list)

        to_use_sets_array=Parallel(n_jobs=GLOBAL_PARALLEL_N_JOBS)(
            delayed(get_all_near)(intervention_set) for intervention_set in exploration_set_list)
        to_calc_sets_array=[]
        for l in to_use_sets_array:
            to_calc_sets_array+=l
        to_calc_sets_array=list(set(to_calc_sets_array))            
        results=Parallel(n_jobs=GLOBAL_PARALLEL_N_JOBS)(delayed(cal_a_set)(s) for s in to_calc_sets_array)
        for i,s in enumerate(to_calc_sets_array):
            coverage_dict[s]=results[i]
        if(len(exploration_set_list[0])==len(manu_var)):
            break
        exploration_set_list=Parallel(n_jobs=GLOBAL_PARALLEL_N_JOBS)(
            delayed(get_best_next)(to_use_list) for to_use_list in to_use_sets_array)
        


def get_coverage(weight_dict:Dict,connection,manu_var:list,greedyType:str,method:str="topo"):
    def cal_coverage_R():
        #Past methods, results are the same as topo
        def calc_a_set_R(s):
            def recuFunc(i:int):
                if is_actived[i] == 1:#If activated
                    return weight_dict[i]
                if in_degree[i] == 0:#In degree is zero, so the contribution is 0
                    return 0
                parents = []
                for idx in np.where(connection[:,i])[0]: # Find parents
                    parents.append(recuFunc(idx))
                return np.mean(parents)
            is_actived = np.zeros((connection.shape[1],)) 
            for si in s:#Iterate all s
                is_actived[si] = 1 #
            return recuFunc(connection.shape[0]-1)
        in_degree=calc_inDegree(connection)
        return calc_a_set_R
    
    def cal_coverage_topo():
        fromList=[]
        for i in range(connection.shape[0]):
            fromList.append([])
        for j in range(connection.shape[0]):
            for i in range(connection.shape[1]):
                if(connection[j,i]):#for all j->i
                    fromList[i].append(j)#Store j in fromList of i
        fromList=fromList
        def calc_a_set_topo(s:interv_set):
            number_of_variables=connection.shape[0]
            is_intervened=np.zeros(number_of_variables,dtype=bool)
            scores=np.zeros(number_of_variables,dtype=float)
            for si in s:
                is_intervened[si]=True
            for i in range(number_of_variables):# for all nodes according to topo order
                if(is_intervened[i]):
                    scores[i]=weight_dict[i]
                elif(len(fromList[i])==0):
                    scores[i]=0 
                else:
                    for j in fromList[i]:
                        scores[i]+=scores[j]
                    scores[i]/=len(fromList[i])
            return scores[number_of_variables-1]
        return calc_a_set_topo
    
    coverage_dict={}
    
    if(method=='topo'):
        cal_a_set=cal_coverage_topo()
    elif(method=='R'):
        cal_a_set=cal_coverage_R()
    else:
        raise BaseException("Illegal method")
    
    if(greedyType=='3'):
        get_K_greed(manu_var=manu_var,cal_a_set=cal_a_set,coverage_dict=coverage_dict,K=3)
    else:
        raise BaseException("Illegal greedyType")
    
    return coverage_dict 