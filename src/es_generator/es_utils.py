from .coverage_generator import *
def get_interventional_bound(bound_mismatch_strategy,minVal,maxVal):
    distance=(maxVal-minVal)
    if(bound_mismatch_strategy=="NO"):
        pass
    elif(bound_mismatch_strategy=="ADD"):
        minVal+=distance*0.3
        maxVal+=distance*0.3
    elif(bound_mismatch_strategy=="COVER"):
        minVal-=distance*0.3
        maxVal+=distance*0.3
    elif(bound_mismatch_strategy=="MIXED"):
        for i in range(minVal.shape[0]):
            if(np.random.rand()>0.5):#possible COVER
                minVal[i]-=distance[i]*0.3
                maxVal[i]+=distance[i]*0.3
    else:
        raise ValueError("bound_mismatch_type not supported")
    min_max_bound=torch.tensor(
            [minVal,
            maxVal]
        )
    return min_max_bound


def get_ECIS_generator(sem_simulative,coverage_strategy="3"):
    ESgenerator=ESmodule(sem_simulative)
    weight_dict,data_x_y_var_dict=get_weight_and_data_x_y_var_dict(sem_simulative,20,7)
    ESgenerator.set_weight_dict(weight_dict)
    ESgenerator.set_x_y_history(data_x_y_var_dict)
    coverage_type_dict={}
    coverage_type_dict[coverage_strategy]=get_coverage(
        ESgenerator.weight_dict
        ,sem_simulative.get_connection()
        ,list(sem_simulative.get_intervenable_variables())
        ,greedyType=coverage_strategy
    )
    #print("For coverage strategy: ",coverage_strategy)
    #print(coverage_type_dict[coverage_strategy])
    ESgenerator.set_covreage_type_dict(coverage_type_dict)
    return ESgenerator
