from .Ablation import run_HCBO_baseline
from ..sem.sem_synt import get_synt_intervention_data,SEM_synt
from .file_analysis import ResultManager
from ..utils import *
from ..cbo.CommonExperiment import optimize_with_CBO
import traceback

def combine_HCBO_weight_data(init_intervention_data,data_x_y_var_dict):
    for variable,x_y_list in data_x_y_var_dict.items():
        intervention_set=(variable,)
        if(intervention_set in init_intervention_data):
            print("Loading x and y used to calculate weight")
            weight_data_x,weight_data_y=x_y_list
            weight_data_x=weight_data_x.tolist()
            weight_data_y=weight_data_y.tolist()
            exist_x,exist_y =init_intervention_data[intervention_set]
            for i,(x,y) in enumerate(zip(weight_data_x,weight_data_y)):
                exist_x.append(x)
                exist_y.append(y)
            init_intervention_data[intervention_set]=[exist_x,exist_y]
        else:
            init_intervention_data[intervention_set]=x_y_list
    return init_intervention_data
def run_random_baseline(
    target_sem:SEM_synt,optimization_cost,**kwargs
):
    result_x_list=[]
    result_y_list=[]
    visit_IS_list=[]
    cumulative_cost=0
    intervenable_variables=target_sem.get_intervenable_variables()
    while(cumulative_cost<=optimization_cost):
        print(cumulative_cost)
        subset_length=random.randint(1,len(intervenable_variables))
        intervention_subset = tuple(sorted(random.sample(intervenable_variables, subset_length))) 
        bounds=target_sem.get_bounds(intervention_subset)
        intervention_plan=get_random_intervention_plan(
            intervention_set=intervention_subset,
            min_vals=bounds[0],
            max_vals=bounds[1]
        )
        y_val=target_sem.intervene(1,SEM_REPEATED_TIMES,intervention_plan)
        now_x=[x for i,x in intervention_plan]
        result_x_list.append(now_x)
        result_y_list.append(y_val[0])
        visit_IS_list.append(intervention_subset)
        cumulative_cost+=len(intervention_subset)
    return result_x_list,result_y_list,visit_IS_list,{
            "Set_Selector_Type":"Random",
            "acq_history":[],
            "mean_history":[],
            "issf_history":[]
        }
def run_CBO_baseline(
        target_sem,init_intervention_data,optimization_cost,full_observational_samples,**kwargs
    ):
        return optimize_with_CBO(
            target_sem,
            intervention_data_dict=init_intervention_data,
            full_observational_samples=full_observational_samples,
            optimization_cost=optimization_cost,
            es_hcbo_form=target_sem.MIS,
            Causal_prior=False,
        )

def causal_run(
        target_sem,init_synt_data_per_subset,run_CBO,
        name,hcbo_lowd_name,es,run_function,ECCIS_limit10,
        optimization_cost,full_observational_samples,
        data_x_y_var_dict,result_saver,is_linear,repeated_times
    ):
    succ_times=0
    for seed in range(MAX_TRY_TIME):
        init_context(seed)
        try:
            if(run_CBO):
                init_intervention_data=get_synt_intervention_data(
                    target_sem,
                    get_power_set_without_empty(target_sem.get_intervenable_variables()),
                    init_synt_data_per_subset
                )
            else:
                init_intervention_data=get_synt_intervention_data(
                    target_sem,
                    es,
                    init_synt_data_per_subset
                )
            if(name=='HCBO' or name == hcbo_lowd_name):
                init_intervention_data=combine_HCBO_weight_data(init_intervention_data,data_x_y_var_dict=data_x_y_var_dict)
            
            print(f"Begin to run {name} with seed as {seed}")
            result=run_function(
                    target_sem=target_sem,
                    es=es if name!=hcbo_lowd_name else ECCIS_limit10,
                    init_intervention_data=init_intervention_data,
                    optimization_cost=optimization_cost,
                    full_observational_samples=full_observational_samples,
                    is_linear=is_linear
                )
            
            result_saver.save_causal_baseline_result(name,result,seed)
        except Exception as e:
            print(e)
            continue
        succ_times+=1
        if(succ_times>=repeated_times):
            break
class HCBO_BaselineRunner():
    def __init__(self,target_sem,es,optimization_cost,is_linear,x_y_history,pool):
        self.target_sem=target_sem
        self.es=es
        self.optimization_cost=optimization_cost
        self.is_linear=is_linear
        self.data_x_y_var_dict=x_y_history
        self.pool=pool
    def run_all(self,result_saver:ResultManager,repeated_times,run_CBO=False,full_observational_samples=None,ECCIS_limit10=None):
        if(run_CBO):
            init_synt_data_per_subset=3
        else:
            init_synt_data_per_subset=SYNT_INIT_INTERVENTION_DATA_NUM
        hcbo_lowd_name="HCBO-10"
        name_function_list=[
            ("HCBO",run_HCBO_baseline),
            ("Random-Search",run_random_baseline),
        ]
        if(run_CBO):
            name_function_list.append(
                ("CBO",run_CBO_baseline)
            )
        result_dict={}
        for name,run_function in name_function_list:
            result_dict[name]=self.pool.apply_async(
                func=causal_run
                ,args=(
                self.target_sem,init_synt_data_per_subset,run_CBO,
                name,hcbo_lowd_name,self.es,run_function,ECCIS_limit10,
                self.optimization_cost,full_observational_samples,
                self.data_x_y_var_dict,result_saver,self.is_linear,repeated_times
                )
            )
        return result_dict

class HCBO_Fitted_BaselineRunner():
    def __init__(self,target_sem,es,optimization_cost,is_linear,x_y_history,pool):
        self.target_sem=target_sem
        self.es=es
        self.optimization_cost=optimization_cost
        self.is_linear=is_linear
        self.data_x_y_var_dict=x_y_history
        self.pool=pool
    def run_all(self,result_saver:ResultManager,repeated_times,run_CBO=False,full_observational_samples=None,ECCIS_limit10=None):
        if(run_CBO):
            init_synt_data_per_subset=3
        else:
            init_synt_data_per_subset=SYNT_INIT_INTERVENTION_DATA_NUM
        hcbo_lowd_name="HCBO-10"
        name_function_list=[
            ("HCBO-ECCIS-fitted",run_HCBO_baseline),
        ]
        result_dict={}
        for name,run_function in name_function_list:
            result_dict[name]=self.pool.apply_async(
                func=causal_run
                ,args=(
                self.target_sem,init_synt_data_per_subset,run_CBO,
                name,hcbo_lowd_name,self.es,run_function,ECCIS_limit10,
                self.optimization_cost,full_observational_samples,
                self.data_x_y_var_dict,result_saver,self.is_linear,repeated_times
                )
            )
        return result_dict





"""
class CBO_BaselineRunner():
    def __init__(self,graph_name:str):
        self.target_sem=target_sem
        self.es=es
        self.optimization_cost=optimization_cost
    def run_all(self,result_saver:ResultManager,repeated_times):
        name='HCBO'
        def run_with_seed(seed):
            init_context(seed)
            try:
                return optimize_with_CBO(
                    target_sem=self.target_sem,
                    es=self.es,
                    init_intervention_data=get_synt_intervention_data(
                        self.target_sem,
                        self.es
                    ),
                    optimization_cost=self.optimization_cost
                )
            except Exception as e:
                print(e)
                return []
        results=Parallel(n_jobs=repeated_times)(
            delayed(run_with_seed)(t) for t in range(repeated_times)
        )
        result_saver.save_causal_baseline_result(name,results)
"""