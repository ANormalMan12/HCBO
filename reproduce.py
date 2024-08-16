import subprocess
import pathlib
import multiprocessing
import os
from datetime import datetime

def get_next_gpu_id():
    return None
def get_time_str():
    current_time = datetime.now()
    time_string = current_time.strftime(f"%Y-%m-%d-%H-%M-%S")
    return time_string

LOG_DIR_PATH=pathlib.Path("logs")
import copy
def get_synt_name(dim:int,seed:int,func_type:str):
    return func_type+'-'+str(dim)+'-'+str(seed)
def run_python(script_name:str,args:list,log_dir_name:str,log_file_name:str,env=None):
    if(env is not None):
        sys_env=copy.deepcopy(os.environ)
        for key,value in env.items():
            sys_env[key]=value
        env=sys_env
    os.makedirs(LOG_DIR_PATH/log_dir_name,exist_ok=True)
    with open(LOG_DIR_PATH/log_dir_name/(log_file_name+".log"),"w") as logf:
        subprocess.check_call(['python',script_name]+args,stdout=logf,env=env)

def initialize_synt_data(dim:int,seed:int,func_type:str):
    save_name=get_synt_name(dim,seed=seed,func_type=func_type)
    args=[str(dim),str(seed),save_name,'--random_function_strategy',func_type]
    run_python('initialize_synt.py',args,"init",save_name)

def initialize_real_data(graph_name):
    run_python('initialize_real.py',[graph_name],"init",graph_name)



class InitializationClass():
    def __init__(self,synt_dataset_list,real_dataset_list,pool_size=4):
        self.synt_dataset_list=synt_dataset_list
        self.real_dataset_list=real_dataset_list
        self.experiment_name_list=[]
        self.pool_size=pool_size
    def initialize_data(self):
        experiment_name_list=[]
        with multiprocessing.Pool(processes=self.pool_size) as p:
            results=[]
            for dim,seed,func_type in self.synt_dataset_list:
                save_name=get_synt_name(dim,seed,func_type)
                results.append(p.apply_async(
                    func=initialize_synt_data,
                    args=(dim,seed,func_type)
                ))
                print("Submit:",save_name)
                experiment_name_list.append(save_name)
            for graph_name in self.real_dataset_list:
                results.append(p.apply_async(
                    func=initialize_real_data,
                    args=(graph_name,)
                ))
                print("Submit:",graph_name)
                experiment_name_list.append(graph_name)
            for result in results:
                print(result.get())
            for i,result in enumerate(results):
                if not result.successful():
                    print("Fail in ",experiment_name_list[i])
                    experiment_name_list[i]+="-(Fail)"
            
        self.experiment_name_list=experiment_name_list
        print("Submission Done:",self.experiment_name_list)
        init_log_dir=LOG_DIR_PATH/"init"
        os.makedirs(init_log_dir,exist_ok=True)
        with open(init_log_dir/(get_time_str()+".txt"),"w") as f:
            f.write("\n".join(experiment_name_list))
        return experiment_name_list
    
class RunClass():
    def __init__(self,experiment_name,gpu_device_id):
        self.experiment_name=experiment_name
        self.gpu_device_id=gpu_device_id

    def _run(self,action_name:str,is_try_mode:bool=False):
        if(self.gpu_device_id==None):
            env_vars=None
        else:
            env_vars = {'CUDA_VISIBLE_DEVICES': str(self.gpu_device_id)}
        arg_python_list=[self.experiment_name,"--run_"+action_name]
        if(is_try_mode):
            arg_python_list+=['--is_try_mode']    
        run_python('run.py',arg_python_list,action_name,self.experiment_name,env=env_vars)
        try:
            os.makedirs(LOG_DIR_PATH/action_name,exist_ok=True)
            with open(LOG_DIR_PATH/action_name/(get_time_str()+".txt"),"w") as f:
                f.write("\n".join(self.experiment_name))
        except Exception as e:
            print(e)
    def run_ablation(self):
        self._run("ablation")
    def run_hyperparameter(self):
        self._run("hyperparameter")
    def run_performance(self,is_try_mode):
        self._run("performance",is_try_mode=is_try_mode)


def run_one_analysis(experiment_name):
    run_instance=RunClass(experiment_name=experiment_name,gpu_device_id=get_next_gpu_id())
    run_instance.run_performance(False)
    run_instance.run_hyperparameter()
    run_instance.run_ablation()
def run_one_baseline(experiment_name):
    run_instance=RunClass(experiment_name=experiment_name,gpu_device_id=get_next_gpu_id())
    run_instance.run_performance(False)
def run_try_baseline(experiment_name):
    run_instance=RunClass(experiment_name=experiment_name,gpu_device_id=get_next_gpu_id())
    run_instance.run_performance(True)

def run_all_baselines(need_initialize,is_try):
    synt_dataset_list=[
        [50,5,"non-additive"],
        [50,122,"non-additive"],
        [50,123,"non-additive"],
    ] 
    real_dataset_list=[
        #"CoralGraph"
    ]
    if(need_initialize):
        experiment_list=InitializationClass(synt_dataset_list,real_dataset_list,pool_size=2).initialize_data()
    else:
        experiment_list=[get_synt_name(*arg) for arg in synt_dataset_list]
        experiment_list+=real_dataset_list
    if(is_try):
        run_HCBO_func=run_try_baseline
    else:
        run_HCBO_func=run_one_baseline
    with multiprocessing.Pool(processes=1) as p:
        results=[]
        for exp_name in experiment_list:
            print(exp_name)
            results.append(
                p.apply_async(
                func=run_HCBO_func,
                args=(exp_name,)
            ))
        for result in results:
            print(result.get())
            if not result.successful():
                print("Fail")



if(__name__=='__main__'):    
    synt_dataset_list=[
        [50,124,"additive"],
        [100,8,"additive"],
        [100,124,"linear"],
        [200,2,"linear"],
        [50,122,"non-additive"],
        [100,124,"non-additive"],
    ]
    real_dataset_list=[
        #"CoralGraph"
    ]
    exp_name_list=InitializationClass(synt_dataset_list,real_dataset_list,2).initialize_data()