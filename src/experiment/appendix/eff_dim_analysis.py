from src import *
from ..hdo_baselines import optimize_with_TuRBO,get_hdo_from_hco


def parallel_sub_space_sample(causal_intervention_set,target_scm:SEM,maximize,optimization_iteration):
    init_context(0)
    causal_intervention_set=tuple(sorted(causal_intervention_set))
    problem=get_hdo_from_hco(target_scm,causal_intervention_set)
    X,fX=optimize_with_TuRBO(problem,optimization_iteration)
    if(maximize):
        return np.max(fX)
    return np.min(fX)
def analyze_effective_dimension(target_scm:SEM_synt,oracle_eccis,sample_times_per_dimension,optimization_iteration):  
    manu_vars=tuple(target_scm.get_intervenable_variables())
    maximize=(target_scm.task=='max')
    result_dict={}
    N=len(manu_vars)
    for n_dim in range(1,len(manu_vars)+1):
        print(f"Begin to analyze dimension {n_dim}")
        intervention_set_sample_list=[]
        intervention_set_sample_list.append(tuple(oracle_eccis[n_dim-1]))
        
        number_all_subsets=math.comb(N,n_dim)
        if(number_all_subsets<=sample_times_per_dimension):
            intervention_set_sample_list=generate_sorted_K_combinations(manu_vars,n_dim)
        else:
            for i_sub_space in range(sample_times_per_dimension):
                intervention_set_sample_list.append(tuple(sorted(
                    random.sample(list(target_scm.get_intervenable_variables()),n_dim))))
            intervention_set_sample_list=list(set(intervention_set_sample_list))
        print(f"{len(intervention_set_sample_list)} sampled intervention sets from dimension {n_dim}")
        best_value_list=Parallel(n_jobs=min(2,len(intervention_set_sample_list)))(
            delayed(parallel_sub_space_sample)
            (causal_intervention_set=intervention_set,target_scm=target_scm,maximize=maximize,optimization_iteration=optimization_iteration)
            for intervention_set in intervention_set_sample_list
        )
        assert(len(intervention_set_sample_list)==len(best_value_list))
        best_intervention_set=None
        best_value=-np.inf if maximize else np.inf
        
        for intervention_set,good_value in zip(intervention_set_sample_list, best_value_list):
            if(number_all_subsets<=sample_times_per_dimension):
                print(intervention_set,":",good_value)
            if(maximize):
                if(good_value>best_value):
                    best_intervention_set=intervention_set
                    best_value=good_value
            else:
                if(good_value<best_value):
                    best_intervention_set=intervention_set
                    best_value=good_value
        result_dict[n_dim]={"best_intervention_set":best_intervention_set,"best_value":best_value,"number_sampled_intervention_set":len(intervention_set_sample_list)}
    return pd.DataFrame(result_dict)
def eff_dim_I_only(optimization_iteration):
    import argparse
    parser = argparse.ArgumentParser(description="Effective Dimension Analysis")
    parser.add_argument("exp_name",type=str)
    #parser.add_argument("--seed",default=0,type=int)
    args=parser.parse_args()
    exp_data_reader=ExperimentDataReader(args.exp_name)
    target_scm=exp_data_reader.read_oracle_sem()
    I=target_scm.get_intervenable_variables()
    val=parallel_sub_space_sample(
        causal_intervention_set=I,
        target_scm=target_scm,
        maximize=(target_scm.task=="max"),
        optimization_iteration=optimization_iteration
    )
    pathlib.Path(RESULT_DIR_PATH/"EffDim").mkdir(parents=True,exist_ok=True)
    with open(RESULT_DIR_PATH/"EffDim"/(args.exp_name+f"-I-{optimization_iteration}.txt"),"w") as f:
        print(val,f)

def eff_dim_main():
    import argparse
    parser = argparse.ArgumentParser(description="Effective Dimension Analysis")
    parser.add_argument("exp_name",type=str)
    #parser.add_argument("--seed",default=0,type=int)
    args=parser.parse_args()
    exp_data_reader=ExperimentDataReader(args.exp_name)
    oracle_sem=exp_data_reader.read_oracle_sem()
    oracle_eccis=exp_data_reader.read_oracle_ECCIS()
    if(args.exp_name=="CoralGraph" or args.exp_name=="HealthGraph"):
        optimization_iteration=1000
        most_sampled_subset_number=100
    else:
        optimization_iteration=500
        most_sampled_subset_number=10
    pathlib.Path(RESULT_DIR_PATH/"EffDim").mkdir(parents=True,exist_ok=True)
    df=analyze_effective_dimension(oracle_sem,oracle_eccis,most_sampled_subset_number,optimization_iteration)
    csv_name=RESULT_DIR_PATH/"EffDim"/(args.exp_name+"-100.csv")
    df.to_csv(csv_name)
