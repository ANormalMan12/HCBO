from .utils import *

def analyze_hcbo_results(all_data_dict,maximize):
    covergence_data_dict={}
    issf_info_list_dict={}
    report_info={}
    for name,hcbo_return_group in all_data_dict.items():
        print(f"---{name}---")
        x_y_array=[]
        issf_info_list=[]
        for run in hcbo_return_group:
            if(len(run)==0):
                print("Detect One Failed Run in ",name)
                continue
            result_x_list,result_y_list,visit_IS_list,issf_info=run
            x_y_array.append((result_x_list,result_y_list,visit_IS_list))
            issf_info_list.append(issf_info)
        if(len(x_y_array)==0):
            print("No valid data for ",name)
            continue
        best_y_per_cost_matrix,best_val,best_x,best_IS,best_mean,best_std=get_bestPERcostMatrix_bestx_besty_bestdim_best_mean_std(x_y_array,maximize)
        report_info[name]={
            "best_Y":best_val,
            "best_intervention_set":best_IS,
            "best_x":best_x,
            "best_dim":len(best_IS),
            "best_mean":best_mean,
            "best_std":best_std
        }
        covergence_data_dict[name]=best_y_per_cost_matrix
        issf_info_list_dict[name]=issf_info_list
    return covergence_data_dict,report_info,issf_info_list_dict

