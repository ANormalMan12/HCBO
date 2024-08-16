from .co_analysis import get_bestPERcostMatrix_bestx_besty_bestdim_best_mean_std
def analyze_hdo_results(all_x_y_per_iteration_data_dict,maximize):
    covergence_data_dict={}
    for name,hdo_x_y_group in all_x_y_per_iteration_data_dict.items():
        print(f"---{name}---")
        
        x_y_is_list=[]
        for x_y_list in hdo_x_y_group:
            if(len(x_y_list)==0):
                raise ValueError("Problems")
            x_y_is_list.append([
                x_y_list[0],
                x_y_list[1],
                [None for i in range(len(x_y_list[1]))]
            ])
        if(len(x_y_list) == 0 ):
            print("All failed")
            continue
        best_y_per_cost_matrix,best_val,best_x,best_IS,best_mean,best_std=get_bestPERcostMatrix_bestx_besty_bestdim_best_mean_std(
            x_y_is_list,
            maximize
        )
        print(best_x,best_val)
        covergence_data_dict[name]=best_y_per_cost_matrix
    
    return covergence_data_dict