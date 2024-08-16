from src.utils import *
from src.sem import *
from src.es_generator import *
from src.experiment import *
import argparse


def get_random_SEM_topo(d,prob_edge,seed):
    connection=get_random_DAG(d,prob_edge,seed)
    manu_vars=random.sample(range(d-1),d//3)
    return connection,manu_vars
def get_random_DAG(d,prob_edge,seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    connection=np.zeros((d,d),dtype=bool)
    for i in range(d-2,-1,-1):#d-2,d-3,...,1,0
        connection[i,random.randint(i+1,d-1)]=True
        for j in range(i+1,d,1):
            if(np.random.rand()<prob_edge):
                connection[i,j]=True
    return connection
from src.sem.node import *
def get_random_node_function_builder_list(
    connection,
    pars_list,
    function_type_prob_dict,
    use_gp:bool=False
):
    
    _funciton_builder_list=[]
    for i in range(connection.shape[0]):
        func_type_list=list(function_type_prob_dict.keys())
        p_list=list(function_type_prob_dict.values())
        function_type=np.random.choice(
            a=func_type_list,
            p=p_list
        )
        if(use_gp):
            node_builder=NodeFunctionBuilderGP(len(pars_list[i]),function_type)
        else:  
            node_builder=NodeFunctionBuilder(len(pars_list[i]),function_type)
        _funciton_builder_list.append(node_builder)
    return _funciton_builder_list
    

def init_sem_min_max(sem,bound_mismatch_type):
    sample_data=sem.sample(n_samples=300)
    minVal=np.min(sample_data,axis=0)
    maxVal=np.max(sample_data,axis=0)
    sem.set_min_max_bounds(get_interventional_bound(bound_mismatch_type,minVal,maxVal))
    return sem
def get_synt_builder_sem(d,prob_edge,seed,random_function_strategy,bound_mismatch_type="MIXED"):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    #W,connectionT,manu_vars= select_W_adjMatrix(d=d,prob_edge=prob_edge,seed=seed)
    connection,manu_vars=get_random_SEM_topo(d,prob_edge,seed)
    if(random_function_strategy=="linear"):
        function_type_prob_dict={
            "linear":1.0
        }
    elif(random_function_strategy=="additive"):
        function_type_prob_dict={
            "additive":1.0
        }
    elif(random_function_strategy=="non-additive"):
        function_type_prob_dict={
            "additive":0.5,
            "non-additive":0.5
        }
    else:
        raise ValueError("random_function_strategy not supported")

    init_builder_list:List[NodeFunctionBuilder]=get_random_node_function_builder_list(
        connection=connection,
        pars_list=get_pars_list(connection),
        function_type_prob_dict=function_type_prob_dict
    )

    oracle_function_list=[builder.get_oracle_node() for builder in init_builder_list]
    
    the_sem=SEM_synt(
        name=random_function_strategy+'-'+str(d)+'-'+str(int(prob_edge*100))+"-"+str(seed)+("-mismatch_"+bound_mismatch_type)+'-'+get_time(),
        connectionT=connection.T,
        intervenable_variable_list=sorted(list(manu_vars)),
        noise_type='gaussian',
        task='max',
        pars_list=get_pars_list(connection),
        function_node_list=oracle_function_list
    )
    the_sem=init_sem_min_max(the_sem,bound_mismatch_type)
    return (init_builder_list,the_sem)

def get_fitted_sem_and_save(oracle_sem:SEM_synt,node_function_builder_list,experiment_dir_path):
    save_pickle(
        oracle_sem,
        experiment_dir_path/ORACLE_SEM_NAME
    )
    sample_np_data=oracle_sem.sample(n_samples=FITTED_SAMPLE_NUM)
    sample_df_data=pd.DataFrame(sample_np_data)

    if(hasattr(oracle_sem,"num_name")):
        num_name=oracle_sem.num_name
    else:
        num_name=[i for i in range(oracle_sem._connection.shape[0])]
    sample_df_data.columns=num_name
    sample_df_data.to_csv(experiment_dir_path/(str(FITTED_SAMPLE_NUM)+"-sample_data_to_fit.csv"),index=False)
    fitted_sem:SEM_synt=get_fitted_sem(
        connection=oracle_sem._connection,
        num_name=num_name,
        manu_variable_list=oracle_sem.get_intervenable_variables(),
        true_observation=sample_df_data,
        sem_name=oracle_sem.name+"-fitted-"+str(FITTED_SAMPLE_NUM),
        function_builder_list=node_function_builder_list,
        bounds=oracle_sem.all_variable_space_torch,
        task=oracle_sem.task
    )
    save_pickle(
        fitted_sem,
        experiment_dir_path/FITTED_SEM_NAME
    )
    return fitted_sem

def initialize_HCBO_related(node_function_builder_list,oracle_sem,save_name):
    experiment_dir_path=GRAPH_DIR_PATH/save_name
    fitted_sem=get_fitted_sem_and_save(oracle_sem,node_function_builder_list,experiment_dir_path)

    oracle_es_module:ESmodule=get_ECIS_generator(oracle_sem)
    save_pickle(
        oracle_es_module,
        experiment_dir_path/"oracle_ECIS_generator.pkl"
    )
    oracle_ECCIS=get_bestES_per_dimension(oracle_sem.get_intervenable_variables(),oracle_es_module.coverage_type_dict["3"])
    save_json({"oracle_ECCIS":oracle_ECCIS},experiment_dir_path/"oracle_ECCIS.json")

    #fitted_es_module:ESmodule=get_ECIS_generator(fitted_sem)
    #save_pickle(
    #    fitted_es_module,
    #    experiment_dir_path/"fitted_ECIS_generator.pkl"
    #)
    #fitted_ECCIS=get_bestES_per_dimension(fitted_sem.get_intervenable_variables(),fitted_es_module.coverage_type_dict["3"])
    #save_json({"fitted_ECCIS":fitted_ECCIS},experiment_dir_path/"fitted_ECCIS.json")
