from .draw_utils import *
def issf_analysis(name,ISSF_info):
    acq_history=ISSF_info["acq_history"]
    mean_history=ISSF_info["mean_history"]
    issf_history=ISSF_info["issf_history"]
    fig,axes=plt.subplots(3,2)
    if(acq_history is not None and len(acq_history)>0):
        sns.heatmap(acq_history,ax=axes[0,0])
        sns.heatmap(val_history2rank_history(acq_history),ax=axes[0,1])
        axes[0,0].set_title("Acquisition Value")
        axes[0,1].set_title("Acquisition Ranks")
    if(mean_history is not None and len(mean_history)>0):
        sns.heatmap(mean_history,ax=axes[1,0])
        sns.heatmap(val_history2rank_history(mean_history),ax=axes[1,1])
        axes[1,0].set_title("Mean Statistic of y")
        axes[1,1].set_title("Mean Statistic of y (Rank)")
    if(issf_history is not None and len(issf_history)>0):
        sns.heatmap(issf_history,ax=axes[2,0])
        sns.heatmap(val_history2rank_history(issf_history),ax=axes[2,1])
        axes[2,0].set_title("ISSF Values")
        axes[2,1].set_title("ISSF Ranks")
    fig.suptitle(name)
    plt.tight_layout()
    return fig
def get_bar_by_visit_history(visit_IS_index_list):
    IS_numbers=-np.inf
    for IS_index in visit_IS_index_list:
        IS_numbers=max(IS_index,IS_numbers)
    bar_lengths=[0 for i in range(IS_numbers+1)]
    for IS_index in visit_IS_index_list:
        bar_lengths[IS_index]+=1
    return bar_lengths
    
def rank_elements(lst):
    indexed_lst = [(value, index) for index, value in enumerate(lst)]
    sorted_lst = sorted(indexed_lst)#Smalller means the rank is higher
    rankings = {}
    for rank, (value, index) in enumerate(sorted_lst):
        rankings[index] = rank + 1
    ranks = [rankings[index] for _, index in indexed_lst]
    return ranks

def val_history2rank_history(val_history):
    rank_history=[
        rank_elements(val_list)
        for val_list in val_history
    ]
    return rank_history