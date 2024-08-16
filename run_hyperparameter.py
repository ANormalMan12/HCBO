from src.init_lib import *
import argparse
from multiprocessing import set_start_method
def get_args_for_run():
    parser = argparse.ArgumentParser(description="Run HCBO")
    parser.add_argument('experiment_name',type=str)
    parser.add_argument('--process_number',default=4, type=int)
    parser.add_argument('--init_gpu_index',default=0, type=int) 
    return parser.parse_args()

def get_hcbo_report(result_saver:ResultManager,name,maximize):
    data_dict={"HCBO":result_saver.read_causal_baseline_result(name)}
    covergence_data_dict,report_info,issf_info_list_dict=analyze_hcbo_results(data_dict,maximize)
    return report_info
def real_mp_run(func,args,gpu_index):
    if(torch.cuda.is_available()):
        torch.cuda.set_device(gpu_index)
    func(*args)
if __name__ == '__main__':
    init_context(0)
    
    np.seterr(over='raise')
    args=get_args_for_run()
    if torch.cuda.is_available():
        print("cuda is available")
    else:
        print("use CPU")
    
    data_reader=ExperimentDataReader(args.experiment_name)
    target_sem:SEM_synt=data_reader.read_oracle_sem()
    ECCIS=data_reader.read_oracle_ECCIS()
    
    is_linear=(args.experiment_name[:6]=='linear') or (args.experiment_name[:5]=="Coral")
    if(is_linear):
        print("Use additive kernel for HCBO, BO and dropout-BO")
    else:
        print("Use RBF kernel for HCBO, BO and dropout-BO")
    I_size=len(target_sem.get_intervenable_variables())
    D=target_sem.all_variable_space_torch.shape[1]
    run_CBO=False
    full_observational_samples=None
    ECCIS_limit10=None
    #es_generator:ESmodule=data_reader.read_oracle_coverage_generator()
    x_y_history=data_reader.read_oracle_coverage_generator().x_y_history
    if(D==50):
        cost=3000
    elif(D==100):
        cost=6000
    elif(D==150):
        cost=10000
    elif(D==200):
        cost=17000
    elif(D<20):
        cost=100*I_size
        if(hasattr(target_sem,"MIS")):
            run_CBO=True
            #kv_list= sorted(list(es_generator.coverage_type_dict["3"].items()),key=lambda item:item[1])
            #ECCIS_limit10=[kv[0] for kv in  kv_list[:int(0.5*len(kv_list))]]
            ECCIS_limit10=get_power_set_without_empty(tuple(target_sem.get_intervenable_variables()))
            print(ECCIS_limit10)
            full_observational_samples=data_reader.read_observation_data()
    else:
        raise("Dimension not in the scope")
    
    
    repeated_run_times=5
    ablation_es_numbers=5
    result_saver=ResultManager(args.experiment_name)
    set_start_method("spawn")
    
    class SelfIdentifiedPool():
        def __init__(self,processes) -> None:
            self.pool=torch.multiprocessing.Pool(processes=processes)
            if torch.cuda.is_available():
                self.gpu_numbers = torch.cuda.device_count()
            else:
                self.gpu_numbers = 0
            self.gpu_index=args.init_gpu_index
        def close(self):
            return self.pool.close()
        def join(self):
            self.pool.join()
        def apply_async(self,func,args):
            self.gpu_index=(self.gpu_index+1)%self.gpu_numbers
            return self.pool.apply_async(real_mp_run,(func,args,self.gpu_index))
            
    pool = SelfIdentifiedPool(args.process_number)
    all_async_result_dict={}
    
    init_intervention_data=get_synt_intervention_data(target_sem,ECCIS,SYNT_INIT_INTERVENTION_DATA_NUM)
    init_intervention_data=combine_HCBO_weight_data(init_intervention_data,data_x_y_var_dict=x_y_history)
    all_async_result_dict.update(run_hyper_Issf_UCBn_update_alpha(target_sem,ECCIS,init_intervention_data,cost,repeated_run_times,result_saver,is_linear,pool))
    all_async_result_dict.update(run_hyper_Issf_UCBn_Fixed_Alpha(target_sem,ECCIS,init_intervention_data,cost,repeated_run_times,result_saver,is_linear,pool))
    
    
    pool.close()
    pool.join()
    for key,result in all_async_result_dict.items():    
        print(f"----------------{key}----------------")
        try:
            print(result.get())
        except Exception as e:
            print("Fail in ",key)
            print(e)
            continue
        if not result.successful():
            print("Fail in ",key)
        else:
            print("Yes in ",key)