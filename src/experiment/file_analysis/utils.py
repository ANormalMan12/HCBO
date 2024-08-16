import numpy as np
from typing import List


def get_bestPERcost_bestX_bestIS(x_list,y_list,visit_is_list,maximize)->List:
    cost_list=[len(x) for x in x_list]
    cost_list=np.array(cost_list)
    y_list=np.array(y_list)
    assert(cost_list.shape[0]==y_list.shape[0])
    assert(len(visit_is_list)==y_list.shape[0])
    if(maximize):
        best_iteration=np.argmax(y_list)
        best_y_list=np.maximum.accumulate(y_list)
    else:
        best_iteration=np.argmin(y_list)
        best_y_list=np.minimum.accumulate(y_list)
    assert(y_list[best_iteration]==best_y_list[-1])
    
    best_x=x_list[best_iteration]
    best_is=visit_is_list[best_iteration]
    Ystar_per_cost=[]
    
    for i in range(cost_list.shape[0]):
        for j in range(cost_list[i]):
            Ystar_per_cost.append(best_y_list[i])
    return Ystar_per_cost,best_x,best_is

def get_bestPERcostMatrix_bestx_besty_bestdim_best_mean_std(lists_of_xlist_ylist_ISlist,maximize,minimize_length=False)->np.array:    
    Ystar_cost_matrix=[]
    max_cost=0
    min_cost=np.inf
    best_x=None
    best_IS=None
    if(maximize):
        best_val=-np.inf
    else:
        best_val=np.inf
    best_list=[]
    for x_y_IS_list in lists_of_xlist_ylist_ISlist:
        new_line,new_best_x,new_best_IS=get_bestPERcost_bestX_bestIS(x_y_IS_list[0],x_y_IS_list[1],x_y_IS_list[2],maximize)
        best_list.append(new_line[-1])
        if(maximize):
            if(new_line[-1]>best_val):
                best_val=new_line[-1]
                best_x=new_best_x
                best_IS=new_best_IS
        else:
            if(new_line[-1]<best_val):
                best_val=new_line[-1]
                best_x=new_best_x
                best_IS=new_best_IS
        new_length=len(new_line)
        print(new_length)
        max_cost=max(new_length,max_cost)
        min_cost=min(new_length,min_cost)
        Ystar_cost_matrix.append(new_line)
    if(minimize_length):
        for Ystar_cost in Ystar_cost_matrix:
            Ystar_cost=Ystar_cost[:min_cost]
    else:
        for Ystar_cost in Ystar_cost_matrix:
            to_fill_val=Ystar_cost[-1]
            print("Value to fill",to_fill_val)
            to_fill_num=max_cost - len(Ystar_cost)
            print("Number of elements to fill",to_fill_num)
            for i in range(to_fill_num):
                Ystar_cost.append(to_fill_val)
            print("New length",len(Ystar_cost))
    best_mean= np.mean(np.array(best_list))
    best_std = np.std(np.array(best_list)) 
    return np.array(Ystar_cost_matrix),best_val,best_x,best_IS,best_mean,best_std
