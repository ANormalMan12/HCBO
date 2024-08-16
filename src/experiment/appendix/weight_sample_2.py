from src import *
import argparse


class WeightPlot2():
    def main1():
        exp_path="data/pre_experiments/linear-200-10-124-mismatch_ADD-2024-02-09_21-12-58.pkl"
        oracle_weight_path="data/sem_data/linear-200-10-124-mismatch_ADD-2024-02-09_21-12-58-80-weight.pkl"
        oracle_data=read_pickle("data/sem_data/linear-200-10-124-mismatch_ADD-2024-02-09_21-12-58-80-x_y.pkl")
        #exp_path="data/pre_experiments/linear-100-10-124-mismatch_ADD-2024-02-08_16-03-56.pkl"
        #oracle_weight_path="data/sem_data/linear-100-10-124-mismatch_ADD-2024-02-08_16-03-56-100-weight.pkl"
        #oracle_data=read_pickle("data/sem_data/linear-100-10-124-mismatch_ADD-2024-02-08_16-03-56-100-x_y.pkl")
        task='max'


        pre_exp:PreExperiment=read_pickle(exp_path)
        use_weight=pre_exp.es_generator.weight_dict
        oracle_weight=read_pickle(oracle_weight_path)

    def main():
        parser = argparse.ArgumentParser(description="Run HCBO")
        parser.add_argument('experiment_name',type=str)
        args=parser.parse_args()
        exp_dir_path=GRAPH_DIR_PATH/args.experiment_name
        oracle_sem:SEM_synt=read_pickle(exp_dir_path/ORACLE_SEM_NAME)
        oracle_ECIS:ESmodule=read_pickle(exp_dir_path/"oracle_ECIS.pkl")
        fitted_ECIS:ESmodule=read_pickle(exp_dir_path/"fitted_ECIS.pkl")
        print("Oracle:",oracle_ECIS.weight_dict)
        print("Fitted:",fitted_ECIS.weight_dict)
        compare_weight(fitted_ECIS.weight_dict,oracle_ECIS.weight_dict)
        print("Oracle ECIS/Fitted ECIS:")
        oracle_ECIS=get_bestES_per_dimension(oracle_sem.get_intervenable_variables(),oracle_ECIS.coverage_type_dict["3"])
        fitted_ECIS=get_bestES_per_dimension(oracle_sem.get_intervenable_variables(),fitted_ECIS.coverage_type_dict["3"])
        def calc_set_accuracy(true_set,pred_set):
            right_num=0
            for i in range(len(true_set)):
                if(pred_set[i] in true_set):
                    right_num+=1
            return right_num/len(true_set)

        for i in range(len(oracle_ECIS)):
            print(i,calc_set_accuracy(oracle_ECIS[i],fitted_ECIS[i]))
            #print(oracle_ECIS[i],"/",fitted_ECIS[i])