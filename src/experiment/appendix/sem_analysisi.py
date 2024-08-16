from src import *
def calculate_stats(column):
    min_value = column.min()
    max_value = column.max()
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    mean_value = column.mean()
    
    return pd.Series({'Min': min_value, 'Q1': q1, 'Mean': mean_value,'Q3': q3, 'Max': max_value})

def iterate_all_pre_experiments(folder_path):
    to_analyze_list=[]
    with os.scandir(folder_path) as entries:
        for entry in entries:
            # 
            if entry.is_file():
                print(f"File: {entry.path}")
                to_analyze_list.append(entry.path)
            elif entry.is_dir():
                print(f"Folder: {entry.path} is ignored")
    return to_analyze_list
def analyze_SEM_result():

def analyze_SEM_stability(pre_exp:PreExperiment,pklPath:str):
    ES=random_subsets(pre_exp.sem.get_intervenable_variables())
    for IS in [ES[len(ES)//2],ES[-1]]:
        name=str(IS)
        data_dict={}
        interv_plan=get_random_intervention_plan(
                    IS,
                    pre_exp.sem.get_bounds(IS)[0],
                    pre_exp.sem.get_bounds(IS)[1])
        for repeated_tiems in [1,10,100,200,500,1000]:
            Ys=pre_exp.sem.intervene(
                500,
                repeated_tiems,
                interv_plan    
            )
            data_dict[repeated_tiems]=Ys
        df=pd.DataFrame(data_dict)
        tab_df=df.apply(calculate_stats)
        show_pd_table(tab_df)
        sns.boxplot(data=df)
        plt.title(name+os.path.basename(pklPath))
        plt.show()       
def analyze_all_SEM(analyze_function):
    for pklPath in iterate_all_pre_experiments("data/pre_experiments"):
        print(f"-------{pklPath}-------")
        pre_exp:PreExperiment=read_pickle(pklPath)
        analyze_function(pre_exp,pklPath)
def draw_obs_int_range(
    sample_data,
    title:str,
    left_x,
    right_x,
    fig_height=5,
    fig_width=2,
):
    fig,ax=plt.subplots(figsize=(fig_height,fig_width))
    sns.stripplot(x=sample_data,jitter=0.1,ax=ax)
    rectangles = plt.Rectangle(
        (left_x, -0.1), # (x,y) 
        right_x-left_x, 0.2, # width and height 
        # You can add rotation as well with 'angle'
        alpha=0.2, edgecolor="red", linewidth=3, linestyle='solid')
    ax.add_patch(
        rectangles
    )
    ax.set_title(title)
    return fig,ax

import random
def analyze_SEM_data(pre_exp:PreExperiment,pklPath:str):
    sem:SEM=pre_exp.sem
    sample_data=sem.sample_data
    for i in random.sample(list(sem.get_intervenable_variables()),2):
        column_data=sample_data[:,i]
        bounds=sem.get_bounds((i))
        print(bounds[0],bounds[1])
        fig,ax=draw_obs_int_range(
            column_data,
            f"Distribution of Observational Data node {i}",
            left_x=bounds[0],
            right_x=bounds[1],
            fig_height=5,fig_width=2)
        ax.axis('auto')
        fig.show()
#path="data/pre_experiments/linear-100-10-124-mismatch_ADD-2024-02-08_16-03-56.pkl"
#analyze_SEM_data(read_pickle(path),path)