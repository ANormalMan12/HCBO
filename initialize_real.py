from src.init_lib import *
from src.cbo.graph import *
import bnlearn as bn
import argparse

def get_real_sem(graph_name):
    function_type_prob_dict={#Only for 
        "linear":1.0
    }
    if(graph_name=="CoralGraph"):
        observation_data=pd.read_csv("data/real_data/CoralGraph/observations.csv")
        true_observation=pd.read_csv("data/real_data/CoralGraph/true_observations.csv")
        task='max'
        graph=CoralGraph(
            observation_data,
            true_observation)
        connection=graph.connection
        function_builder_list=get_random_node_function_builder_list(
            connection=connection,
            pars_list=get_pars_list(connection),
            function_type_prob_dict=function_type_prob_dict
        )
    elif(graph_name=="ProteinGraph"):
        true_observation=pd.read_pickle("data/real_data/ProteinGraph/true_observations.pkl")
        def normalize_line(df,name):        
            min_val = df[name].min()
            max_val = df[name].max()
            df[name] = (df[name] - min_val) / (max_val - min_val)
        normalize_line(true_observation,"A")
        task='min'
        graph=ProteinGraph(true_observation)
        connection=graph.connection
        function_builder_list=get_random_node_function_builder_list(
            connection=connection,
            pars_list=get_pars_list(connection),
            function_type_prob_dict=function_type_prob_dict,#Use GP in fact, unused
            use_gp=True
        )
    elif(graph_name=='HealthGraph'):
        healthsem=HealthSEM()
        function_builder_list=get_random_node_function_builder_list(
            healthsem._connection,
            pars_list=get_pars_list(healthsem._connection),
            function_type_prob_dict=function_type_prob_dict,#Use GP in fact, unused
            use_gp=True
        )
        bound_mismatch_type="NO"
        healthsem=init_sem_min_max(healthsem,bound_mismatch_type)
        healthsem.all_variable_space_torch[0,healthsem.CIIndex]=0.0
        healthsem.all_variable_space_torch[1,healthsem.CIIndex]=100.0
        healthsem.all_variable_space_torch[0,healthsem.AspIndex]=0
        healthsem.all_variable_space_torch[1,healthsem.AspIndex]=0.2
        healthsem.all_variable_space_torch[0,healthsem.StaIndex]=0
        healthsem.all_variable_space_torch[1,healthsem.StaIndex]=0.2
        print(healthsem.all_variable_space_torch)
        return function_builder_list,healthsem
    else:
        raise BaseException("Not implemented")
    manu_variable_list=[]
    for s in graph.get_sets()["manu_var"]:
        manu_variable_list.append(graph.num_name.index(s))
    manu_variable_list=sorted(list(manu_variable_list))
    
    intervention_range=graph.get_interventional_ranges()
    bounds=[]
    for i,name in enumerate(graph.num_name):
        if(name=='Y'):continue
        if(i in manu_variable_list):
            bounds.append([
                intervention_range[name][0],
                intervention_range[name][1],
            ])
        else:
            bounds.append([
                float(true_observation[name].min()),
                float(true_observation[name].max())
            ])
        print(name,":",bounds[-1])
    bounds=torch.tensor(bounds).t()

    the_sem=get_fitted_sem(
        connection,
        graph.num_name,
        manu_variable_list,
        true_observation,
        graph_name+get_time(),
        function_builder_list=function_builder_list,
        bounds=bounds,
        task=task
    )
    print(graph.num_name)
    the_sem.num_name=graph.num_name
    if(graph_name=="CoralGraph"):
        the_sem.MIS=get_power_set_without_empty(manu_variable_list)
    elif(graph_name=='ProteinGraph'):
        PKC=0
        PKA=1
        Mek=3
        the_sem.MIS=[
            (PKC,),(PKA,),(Mek,),
            (PKC,PKA),(PKA,Mek),(PKC,Mek)
        ]
    return (function_builder_list,the_sem)


def initialize_real():
    parser = argparse.ArgumentParser(description="Generate data for HCBO")
    parser.add_argument('graph_name',type=str)
    args=parser.parse_args()
    init_context(0)
    function_builder_list,oracle_sem=get_real_sem(args.graph_name)
    initialize_HCBO_related(function_builder_list,oracle_sem,args.graph_name)
    
if(__name__=='__main__'):
    np.seterr(over='raise')
    initialize_real()