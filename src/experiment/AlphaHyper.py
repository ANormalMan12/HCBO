from .Ablation import *
def run_fixed_alpha(target_sem,es,init_intervention_data,optimization_cost,is_linear,result_saver,repeated_times,alpha,i):
    succ_times=0
    for seed in range(MAX_TRY_TIME):
        try:
            init_context(seed)
            result=run_ablation_Issf_fixed_alpha(
                target_sem,es,init_intervention_data,optimization_cost,alpha,is_linear
            )
            result_saver.save_HyperAlpha_Fixed_result(str(i),result,seed)
        except Exception as e:
            print(e)
            continue
        succ_times+=1
        if(succ_times>=repeated_times):
            break
def run_update_alpha(name,run_function,run_args,num_iterations_to_update_alpha,repeated_times,result_saver):
    succ_times=0
    for seed in range(MAX_TRY_TIME):
        try:
            init_context(seed)
            result=run_function(
                    *run_args
                )
            save_name=name+"-"+str(num_iterations_to_update_alpha)
            result_saver.save_HyperAlpha_Updating_result(save_name,result,seed)
        except Exception as e:
            print(e)
            continue
        succ_times+=1
        if(succ_times>=repeated_times):
            break
def run_hyper_Issf_UCBn_update_alpha(target_sem,es,init_intervention_data,optimization_cost,repeated_times,result_saver:ResultManager,is_linear,pool:multiprocessing.Pool):
    result_dict={}
    for num_iterations_to_update_alpha in [1,5,20,50,100]:
        for name,run_function in [
            ("Average",run_ablation_Issf_Average_K_UCBn),
            ("Median",run_ablation_Issf_Median_K_UCBn)]:
                run_args=(target_sem,
                    es,
                    init_intervention_data,
                    optimization_cost,
                    num_iterations_to_update_alpha,
                    is_linear
                )
                result_dict[
                    name+"-"+str(num_iterations_to_update_alpha)
                ]=pool.apply_async(
                    func=run_update_alpha,
                    args=(name,run_function,run_args,num_iterations_to_update_alpha,repeated_times,result_saver)
                )
    return result_dict    


def run_hyper_Issf_UCBn_Fixed_Alpha(target_sem,es,init_intervention_data,optimization_cost,repeated_times,result_saver:ResultManager,is_linear,pool:multiprocessing.Pool):
    result_dict={}
    set_selector=SetSelector_BO(
            lenES=len(es),
            acq_func_type="UCBn",
            issf_strategy=ISSFstrategyAlphaFixed(
                0
            )
        )

    hcbo_optimizer=HCBO_Optimizer(
        set_selector=set_selector,
        val_acqusition_function=find_GP_UCB_acq_val,
        es=es,
        BO_model_class=get_normalized_BO_model_class(is_linear),
        init_intervention_data=init_intervention_data,
        the_sem=target_sem,
        use_global_best_y_for_acq_set=True
    )
    acq_list=set_selector.get_next_acq_list(
        hcbo_optimizer.BO_model_list,
        maximize=hcbo_optimizer.maximize,
        last_update_index=-1,
        global_best_y=hcbo_optimizer.global_best_y
    )
    mean_list=set_selector.get_next_mean_list(
        hcbo_optimizer.BO_model_list,
        last_update_index=-1
    )
    possible_alpha_array=get_possible_alpha_array(mean_list=mean_list,acq_list=acq_list)
    origin_min_alpha=np.min(possible_alpha_array)
    origin_max_alpha=np.max(possible_alpha_array)
    origin_scope=(origin_max_alpha-origin_min_alpha)
    print("Origin Min:",origin_min_alpha)
    print("Origin Max:",origin_max_alpha)
    min_search_bound=max(0,origin_min_alpha-origin_scope/2)
    max_search_bound=origin_max_alpha+origin_scope/2
    print("Min Search Bound:",min_search_bound)
    print("Max Search Bound:",max_search_bound)
    alpha_plans=[np.percentile( possible_alpha_array, p) for p in [25,50,75]]
    result_saver.save_HyperAlpha_plan(alpha_plans)
    for i,alpha in enumerate(alpha_plans):
        result_dict[
            "alpha-"+str(alpha)
        ]=pool.apply_async(
            func=run_fixed_alpha,
            args=(target_sem,es,init_intervention_data,optimization_cost,is_linear,result_saver,repeated_times,alpha,i)
        )
    return result_dict