from .methods.hcbo import *

from .file_analysis import *
def run_ablation_EI_EI(target_sem,es,init_intervention_data,optimization_cost,is_linear):
    return optimize_cgo_with_HCBO(
        target_sem=target_sem,
        set_selector=SetSelector_BO(
            lenES=len(es),
            acq_func_type="EI",
            issf_strategy=ISSFstrategyAcquisition()
        ),
        val_acq_function=find_EI_acq_val,
        BOmodel_class=get_BO_model_class(is_linear),
        es=es,
        init_intervention_data=init_intervention_data,
        optimization_cost=optimization_cost
    )
def run_ablation_UCB_UCB(target_sem,es,init_intervention_data,optimization_cost,is_linear):
    return optimize_cgo_with_HCBO(
        target_sem=target_sem,
        set_selector=SetSelector_BO(
            lenES=len(es),
            acq_func_type="UCB",
            issf_strategy=ISSFstrategyAcquisition()
        ),
        val_acq_function=find_GP_UCB_acq_val,
        BOmodel_class=get_BO_model_class(is_linear),
        es=es,
        init_intervention_data=init_intervention_data,
        optimization_cost=optimization_cost
    )

def run_ablation_Mean_UCBn(target_sem,es,init_intervention_data,optimization_cost,is_linear):
    return optimize_cgo_with_HCBO(
        target_sem=target_sem,
        set_selector=SetSelector_BO(
            lenES=len(es),
            acq_func_type="UCBn",
            issf_strategy=ISSFstrategyMeanY()
        ),
        val_acq_function=find_GP_UCB_acq_val,
        BOmodel_class=get_normalized_BO_model_class(is_linear),
        es=es,
        init_intervention_data=init_intervention_data,
        optimization_cost=optimization_cost
    )
def run_ablation_Issf_Median_K_UCBn(target_sem,es,init_intervention_data,optimization_cost,num_iterations_to_update_alpha,is_linear):
    return optimize_cgo_with_HCBO(
        target_sem=target_sem,
        set_selector=SetSelector_BO(
            lenES=len(es),
            acq_func_type="UCBn",
            issf_strategy=ISSFstrategyAlpha(
                get_alpha_median,times_to_update_alpha=num_iterations_to_update_alpha
            )
        ),
        val_acq_function=find_GP_UCB_acq_val,
        BOmodel_class=get_normalized_BO_model_class(is_linear),
        es=es,
        init_intervention_data=init_intervention_data,
        optimization_cost=optimization_cost
    )
def run_ablation_Issf_fixed_alpha(target_sem,es,init_intervention_data,optimization_cost,alpha,is_linear):
    return optimize_cgo_with_HCBO(
        target_sem=target_sem,
        set_selector=SetSelector_BO(
            lenES=len(es),
            acq_func_type="UCBn",
            issf_strategy=ISSFstrategyAlphaFixed(
                alpha
            )
        ),
        val_acq_function=find_GP_UCB_acq_val,
        BOmodel_class=get_normalized_BO_model_class(is_linear),
        es=es,
        init_intervention_data=init_intervention_data,
        optimization_cost=optimization_cost
    )
def run_ablation_Issf_Average_K_UCBn(target_sem,es,init_intervention_data,optimization_cost,num_iterations_to_update_alpha,is_linear):
    return optimize_cgo_with_HCBO(
        target_sem=target_sem,
        set_selector=SetSelector_BO(
            lenES=len(es),
            acq_func_type="UCBn",
            issf_strategy=ISSFstrategyAlpha(
                get_alpha_average,times_to_update_alpha=num_iterations_to_update_alpha
            )
        ),
        val_acq_function=find_GP_UCB_acq_val,
        BOmodel_class=get_normalized_BO_model_class(is_linear),
        es=es,
        init_intervention_data=init_intervention_data,
        optimization_cost=optimization_cost
    )

def run_ablation_Issf_EIn(target_sem,es,init_intervention_data,optimization_cost,is_linear):
    return optimize_cgo_with_HCBO(
        target_sem=target_sem,
        set_selector=SetSelector_BO(
            lenES=len(es),
            acq_func_type="UCBn",
            issf_strategy=ISSFstrategyAlpha(
                get_alpha_average,times_to_update_alpha=50
            )
        ),
        val_acq_function=find_EI_acq_val,
        BOmodel_class=get_normalized_BO_model_class(is_linear),
        es=es,
        init_intervention_data=init_intervention_data,
        optimization_cost=optimization_cost
    )

def run_HCBO_baseline(
        target_sem,es,init_intervention_data,optimization_cost,is_linear,**kwargs
    ):
    num_iterations_to_update_alpha=50
    return run_ablation_Issf_Average_K_UCBn(
        target_sem,es,init_intervention_data,optimization_cost,num_iterations_to_update_alpha,is_linear
    )

def get_SS_RIS(intervenable_variables):
    subsets = []
    L = len(intervenable_variables)
    for subset_length in range(1, L + 1):
        random_subset = random.sample(intervenable_variables, subset_length)
        subsets.append(tuple(sorted(random_subset)))
    return subsets

def get_DS_RIS(intervenable_variables):
    subsets = []
    L = len(intervenable_variables)
    for k in range(L):
        subset_length=random.randint(1,L)
        random_subset = random.sample(intervenable_variables, subset_length)
        subsets.append(tuple(sorted(random_subset)))
    return subsets
def run_acq_ablation(run_function,repeated_times,target_sem,
                es,
                init_intervention_data,
                optimization_cost,
                is_linear,result_saver,name):
    succ_times=0
    for seed in range(MAX_TRY_TIME):
        try:
            init_context(seed)
            result=run_function(
                target_sem,
                es,
                init_intervention_data,
                optimization_cost,
                is_linear
            )
            result_saver.save_ablation_result(name,result,"Acquisition",seed)
        except Exception as e:
            print(e)
            continue
        succ_times+=1
        if(succ_times>=repeated_times):
            break

def run_es_ablation(target_sem,repeated_times,rand_es,is_linear,result_saver,es_seed,name,optimization_cost):
    succ_times=0
    for seed in range(MAX_TRY_TIME):
        try:
            init_context(seed)
            result=run_HCBO_baseline(
                target_sem,
                es=rand_es,
                init_intervention_data=get_synt_intervention_data(target_sem,rand_es,SYNT_INIT_INTERVENTION_DATA_NUM),
                optimization_cost=optimization_cost,
                is_linear=is_linear
            )
            result_saver.save_ablation_result(name+f"-{es_seed}",result,"ES",seed)
        except Exception as e:
            print(e)
            continue
        succ_times+=1
        if(succ_times>=repeated_times):
            break

class AblationRunner():
    def __init__(self,target_sem,es,init_intervention_data,pool):
        self.target_sem=target_sem
        self.es=es
        self.init_intervention_data=init_intervention_data
        self.pool=pool
    def run_all_acq(self,result_saver:ResultManager,repeated_times,optimization_cost,is_linear):
        results={}
        for (name,run_function) in [
            ("HCBO",run_HCBO_baseline),            
            ("UCB-UCB",run_ablation_UCB_UCB),
            ("ISSF-EIn",run_ablation_Issf_EIn),
            ("Mean-UCBn",run_ablation_Mean_UCBn),
            ("EI-EI",run_ablation_EI_EI),
        ]:
            run_args=(run_function,repeated_times,self.target_sem,
                self.es,
                self.init_intervention_data,
                optimization_cost,
                is_linear,result_saver,name)
            results[name+"-Ablation-Acq"]=self.pool.apply_async(
                func=run_acq_ablation,
                args=run_args
            )
        return results
    def run_all_es(self,result_saver:ResultManager,repeated_times,es_seed_list,optimization_cost,is_linear):
        results={}
        for (name,es_generation_function) in [
            ("SS-RIS",get_SS_RIS),
            ("DS-RIS",get_DS_RIS),
        ]:
            for es_seed in es_seed_list:
                num_cuda_devices = torch.cuda.device_count()
                torch.cuda.set_device((es_seed+random.randint(0,num_cuda_devices))%num_cuda_devices)
                init_context(es_seed)
                rand_es=es_generation_function(self.target_sem.get_intervenable_variables())
                results[name+"-"+str(es_seed)]=self.pool.apply_async(
                    func=run_es_ablation,
                    args=(self.target_sem,repeated_times,rand_es,is_linear,result_saver,es_seed,name,optimization_cost)
                )
        return results
            