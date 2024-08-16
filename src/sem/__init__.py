from .node import *
from .sem_synt import *
from .sem import *
#from .synthetic_datasets import *

def get_fitted_sem(connection,num_name,manu_variable_list,true_observation,sem_name,function_builder_list:List[NodeFunctionBuilder],bounds,task,use_gp:bool=False):
    pars_list=get_pars_list(connection)
    function_node_list=[]
    for node in range(connection.shape[0]):
        node_parents_list=pars_list[node]
        
        print(num_name[node],end=':')
        for par_node in node_parents_list:
            print(num_name[par_node],end=',')
        print()
        
        if(len(node_parents_list)==0):
            function_node_list.append(ZeroNodeFunction())
            continue
        else:
            node_builder:NodeFunctionBuilder=function_builder_list[node]
            input_observation=np.hstack(
                tuple([
                    true_observation[num_name[par_node]][:,np.newaxis] for par_node in node_parents_list 
                ])
            )
            now_node_observation=true_observation[num_name[node]][:,np.newaxis]
            if(use_gp):
                reg_node_function=node_builder.get_fitted_node(input_observation,now_node_observation,"gp")
            else:
                reg_node_function=node_builder.get_fitted_node(input_observation,now_node_observation)
            function_node_list.append(reg_node_function)
    
    theSEM=SEM_synt(
        name=sem_name,
        connectionT=connection.T,
        intervenable_variable_list=manu_variable_list,
        pars_list=pars_list,
        function_node_list=function_node_list,
        task=task
    )
    
    sample_data=[]
    for i in range(connection.shape[0]):
        sample_data.append(true_observation[num_name[i]])
    sample_data=np.array(sample_data).T
    #print(sample_data)

    theSEM.set_min_max_bounds(bounds)
    return theSEM

