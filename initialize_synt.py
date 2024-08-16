
from src.init_lib import *

def initialize_synt():
    parser = argparse.ArgumentParser(description="Generate data for HCBO")
    parser.add_argument('dimension',type=int)
    #parser.add_argument('prob_edge',type=float)
    parser.add_argument('seed',type=int)
    parser.add_argument('save_name',type=str)
    parser.add_argument('--random_function_strategy',default='linear',type=str)
    parser.add_argument("--bound_mismatch_type",default="MIXED",type=str)
    args=parser.parse_args()

    init_context(args.seed)
    (node_function_builder_list,oracle_sem)=get_synt_builder_sem(
        d=args.dimension,
        prob_edge=0.1,
        seed=args.seed,
        random_function_strategy=args.random_function_strategy,
        bound_mismatch_type=args.bound_mismatch_type
    )
    initialize_HCBO_related(node_function_builder_list,oracle_sem,args.save_name)


if(__name__=='__main__'):
    initialize_synt()