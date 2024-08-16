from .methods.hdo_optimizer import *
import copy
from joblib import delayed,Parallel
from .file_analysis import ResultManager
from ..hdo.test_optimization_problem import get_Hartmann6_problem

def real_run_function(target_sem:SEM_synt,name,hd_optimizer,repeated_times,save_manager):
    target_problem=get_hdo_from_hco(target_sem,target_sem.get_intervenable_variables())
    succ_times=0
    for seed in range(MAX_TRY_TIME):
        try:
            init_context(seed)
            result=hd_optimizer(target_problem)
            save_manager.save_hdo_baseline_result(name,result,seed)
        except Exception as e:
            print(e)
            continue
        succ_times+=1
        if(succ_times>=repeated_times):
            break


    
class HDO_BaselineRunner():
    def __init__(self,
                 target_sem:SEM_synt,
                 init_data_num,
                 intrinsic_dimension:int,
                 best_value,
                 is_linear,
                 optimization_cost
                 ):
        bounds=target_sem.get_bounds(target_sem.get_intervenable_variables())
        D=bounds.shape[1]
        self.target_sem=target_sem
        print("Intervenable variables:",D)
        self.init_data_num=init_data_num
        self.optimization_iteration=(optimization_cost//(D+1))
        self.intrinsic_dimension_less_than_20=min(D-1,max(bounds.shape[1]//7,min(20,intrinsic_dimension)))
        self.Cp=best_value*0.05
        self.is_linear=is_linear
    def run_all(self,repeated_times,save_manager:ResultManager,pool):
        result_dict={}
        for (name,run_function) in [
            ("MCTS-VS",MCTSVS_Optimizer(C_p=self.Cp)),
            ("TuRBO",TuRBO_Optimizer()),
            ("CMA-ES",CMAES_Optimizer(0.5)),
            ("Dropout-BO",DropoutBO_Optimizer(
                k=max(2,self.intrinsic_dimension_less_than_20),
                is_add_ucb=self.is_linear
            )),
            ("BO",BO_Optimizer(is_add_ucb=self.is_linear)),
            ("REMBO",REMBO_Optimizer(dim_embedding=self.intrinsic_dimension_less_than_20)),
            ("ALEBO",ALEBO_Optimizer(self.intrinsic_dimension_less_than_20)),
        ]:
            run_function.reset_init_point_budget(self.init_data_num)
            run_function.reset_hdo_optimization_iter(self.optimization_iteration)
            result_dict[name]=pool.apply_async(
                func=real_run_function,
                args=(self.target_sem,name,run_function,repeated_times,save_manager)
            )
        return result_dict

class HDO_Cgo_Optimizer():
    def __init__():
        pass
def optimize_cgo_with_all_hdo(
    target_sem,
    init_data_num,
    optimization_cost,
    intrinsic_dimension:int,
    best_value,
    repeated_times,
    save_manager,
    is_linear,
    pool
):
    
    hdo_runner=HDO_BaselineRunner(
        target_sem,
        init_data_num=init_data_num,
        intrinsic_dimension=intrinsic_dimension,
        best_value=best_value,
        is_linear=is_linear,
        optimization_cost=optimization_cost
    )
    return hdo_runner.run_all(repeated_times,save_manager,pool)

