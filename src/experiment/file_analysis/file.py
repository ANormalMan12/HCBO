import pickle
import os
import json
import numpy as np
import pandas as pd
from ...utils.util_functions import *
class ExperimentDataReader():
    def __init__(self,experiment_name):
        self.experiment_name=experiment_name
        self.experiment_dir=GRAPH_DIR_PATH/experiment_name
    
    def read_oracle_sem(self):
        return read_pickle(self.experiment_dir/ORACLE_SEM_NAME)
    def read_oracle_coverage_generator(self):
        return read_pickle(self.experiment_dir/"oracle_ECIS_generator.pkl")
    
    def read_fitted_ECCIS(self):
        fitted_ECCIS_dict=read_json(self.experiment_dir/"fitted_ECCIS.json")
        fitted_ECCIS=[tuple(IS)for IS in fitted_ECCIS_dict["fitted_ECCIS"]]
        return fitted_ECCIS
    def read_oracle_ECCIS(self):
        oracle_ECCIS_dict=read_json(self.experiment_dir/"oracle_ECCIS.json")
        oracle_ECCIS=[tuple(IS)for IS in oracle_ECCIS_dict["oracle_ECCIS"]]
        return oracle_ECCIS
    
    def read_observation_data(self):
        return pd.read_csv(self.experiment_dir/(str(FITTED_SAMPLE_NUM)+"-sample_data_to_fit.csv")) 
class ResultManager():
    def __init__(self,experiment_name):
        self.experiment_name=experiment_name
        self.result_dir=RESULT_DIR_PATH/experiment_name
        self.hyper_dir=self.result_dir/"HyperAlpha"
        self.hyper_update_dir=self.hyper_dir/"Update"
        self.hyper_fixed_dir=self.hyper_dir/"Fixed"
        self.hyper_fixed_config_path=self.hyper_dir/"config.txt"
    def save_result(self,result_cost_y_index):
        save_pickle(result_cost_y_index,self.result_dir/"result_cost_y_index.pkl")
    
    def save_causal_baseline_result(self,name,results,seed):
        save_pickle(results,self.result_dir/"Baseline"/'Causal'/name/f"{seed}.pkl")
    def save_hdo_baseline_result(self,name,results,seed):
        save_pickle(results,self.result_dir/"Baseline"/'hdo'/name/f"{seed}.pkl")
    def read_causal_baseline_result(self,name):
        results=[]
        for file in (self.result_dir/"Baseline"/'Causal'/(name)).iterdir():
            if file.is_file():
                if(file.name.split(".")[1]!="pkl"):
                    continue
                results.append(read_pickle(file))
        return results
    def read_hdo_baseline_results(self):
        ret_dict={}
        res_dir=self.result_dir/"Baseline"/'hdo'
        for dir in res_dir.iterdir():
            if dir.is_dir():
                key=dir.name.split(".")[0]
                ret_dict[key]=[]
                for file in dir.iterdir():
                    ret_dict[key].append(read_pickle(file))
        return ret_dict
    def save_HyperAlpha_plan(self,plans:np.array):
        if not os.path.exists(os.path.dirname(self.hyper_fixed_config_path)):
            os.makedirs(os.path.dirname(self.hyper_fixed_config_path))
        else:
            raise BaseException("Past HyperAlpha_plan Files should be removed first")
        with open(self.hyper_fixed_config_path,"w") as f:
            np.savetxt(f,plans)
    def save_HyperAlpha_Fixed_result(self,name,results,seed):
        save_pickle(results,self.hyper_fixed_dir/name/f"{seed}.pkl")
    def save_HyperAlpha_Updating_result(self,name,results,seed):
        save_pickle(results,self.hyper_update_dir/name/f"{seed}.pkl")
    def read_HyperAlpha_Updating_results(self):
        ret_dict={}
        for dir in self.hyper_update_dir.iterdir():
            if dir.is_dir():
                key=dir.name.split(".")[0]
                ret_dict[key]=[]
                for file in dir.iterdir():
                    ret_dict[key].append(read_pickle(file))
        return ret_dict
    def read_HyperAlpha_Fixed_results(self):
        ret_dict={}
        txt_path=self.hyper_fixed_config_path
        alpha_list=np.loadtxt(txt_path)
        for dir in self.hyper_fixed_dir.iterdir():
            if dir.is_dir():
                key=dir.name.split(".")[0]
                ret_dict[key]=[]
                for file in dir.iterdir():
                    ret_dict[key].append(read_pickle(file))
        return ret_dict,alpha_list
    def save_ablation_result(self,name,results,subdir_name,seed):
        """All details of Ablation"""
        save_pickle(results,self.result_dir/"Ablation"/subdir_name/name/f"{seed}.pkl")
    def read_ablation_results(self,subdir_name):
        ret_dict={}
        for dir in (self.result_dir/"Ablation"/subdir_name).iterdir():
            if dir.is_dir():
                key=dir.name.split(".")[0]
                ret_dict[key]=[]
                for file in dir.iterdir():
                    ret_dict[key].append(read_pickle(file))
        return ret_dict
    
class FigureSaver():
    def __init__(self,experiment_name):
        self.experiment_name=experiment_name
        self.figure_dir=FIGS_DIR_PATH/experiment_name
    def save_to_ISSF(self,fig,name):
        issf_dir=self.figure_dir/"ISSF_Analysis"
        fig.savefig(issf_dir/name)
    def save_to_Baseline(self,fig,name):
        baseline_dir=self.figure_dir/"Baseline"
        fig.savefig(baseline_dir/name)
    def save_to_Ablation(self,fig,name):
        ablation_dir=self.figure_dir/"Ablation"
        fig.savefig(ablation_dir/name)
    def save_to_Appendix(self,fig,name):
        appendix_dir=self.figure_dir/"Appendix"
        fig.savefig(appendix_dir/name)
def save_text(data,path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print("Save in text to ",path)
    with open(path,"w") as f:
        f.write(data)
def save_pickle(data,path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print("Save in pickle to ",path)
    with open(path,"wb") as f:
        pickle.dump(data,f)
def read_pickle(path):
    print("Read from ", path)
    with open(path,"rb") as f:
        data=pickle.load(f)
    return data
def save_json(data,path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print("Save in json to ",path)
    with open(path,"w") as f:
        json.dump(data,fp=f,indent=4)
def read_json(path):
    print("Read from ", path)
    with open(path,"rb") as f:
        data=json.load(f)
    return data

import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

FITTED_SAMPLE_NUM=300
DATA_DIR_PATH=pathlib.Path("data")
GRAPH_DIR_PATH=DATA_DIR_PATH.joinpath("experiments")
ORACLE_SEM_NAME="oracle_sem.pkl"
FITTED_SEM_NAME=str(FITTED_SAMPLE_NUM)+"-fitted_sem.pkl"
RESULT_DIR_PATH=pathlib.Path("result")
FIGS_DIR_PATH=pathlib.Path("figs")
