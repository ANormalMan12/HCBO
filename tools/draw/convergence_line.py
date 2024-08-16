from .draw_utils import *
import matplotlib.colors as mcolors

def random_initialize_n_m_matrix(n_runs,m_costs,rand_scale:float,function):
    X=np.tile(np.arange(m_costs), (n_runs, 1))
    vec_function=np.vectorize(function)
    Y=vec_function(X)
    rand_change=np.random.uniform(0,rand_scale,size=(n_runs,m_costs))
    return Y+rand_change

def draw_one_line(data,ax:plt.Axes,label,marker=None,color=None):
    """
    data n*m
    """
    means = np.mean(data, axis=0)
    std_devs = np.std(data, axis=0)
    confidence_intervals = 1.96 * (std_devs / np.sqrt(data.shape[0]))  # 95% 
    
    steps = np.arange(0,means.shape[0])

    num_markers = 20
    marker_interval = max(means.shape[0] // num_markers, 1)

    ax.fill_between(steps, means+confidence_intervals, means-confidence_intervals, alpha=0.2,color=color)
    ax.plot(steps, means, label=label,marker=marker,markevery=marker_interval,color=color)

def trans255tofloat(color):
    assert(len(color)==3)
    return([color[i]/255.0 for i in range(3)])


class MyColorClass():
    def __init__(self) -> None:
        self.light_color_dict={
            "green":[142,207,201],
            "orange":[255,190,122],
            "yellow":[237,221,134],
            "red":[250,127,111],
            "blue":[130,176,210],
            "purple":[190,184,220],
            "beige":[231,218,210],
            "black":[153,153,153],
        }
        self.std_dark_color_dict={
            "Blue":(31,119,180),
            "Orange":(255,127,14),
            "Green":(44,160,44),
            "Red":(214,39,40),
            "Purple":(148,103,189),
            "grey":(127,127,127),
            "Pink":(227,119,194),
            "SkyBlue":(23,190,207),
            "brown":(140,86,75),
        }
        self.std_light_color_dict={
            "blue":(174,199,232),
            
            "orange":(255,187,120),
            
            "green":(152,223,138),
            
            "red":(255,152,150),
            
            "purple":(197,176,213),
        }
        self.std_color_dict={
            **self.std_dark_color_dict,
            **self.std_light_color_dict
        }
        self.use_std_color_dict={
            key:trans255tofloat(value) for key,value in self.std_color_dict.items()
        }
    def get_dark_color_list(self):
        return 
    def get_color_list(self):
        return list(self.use_std_color_dict.values())
    def get_color_dict(self):
        return self.use_std_color_dict


def draw_convergence_line_graph(
    subtitle2data_dict:Dict[str,dict],
    xlabel,
    ylabel,
    ylim_bounds=None,
    enable_strize=False,
    subplot_distance_dict:Dict=None
):
    """
    data_dict label_name to n*m data where n means the run round, m means the cost
    """
    color_list= MyColorClass().get_color_list()
    num_sub_fig=len(subtitle2data_dict)
    if(num_sub_fig>3):
        more_3_rows=True
        fig, axes = plt.subplots(2,4,figsize=(3.6*4, 1.6*4))
        #fig, axes = plt.subplots(3,3,figsize=(3.6*4, 2.4*4))
    else:
        more_3_rows=False
        if(num_sub_fig==1):
            fig, axes = plt.subplots(1,num_sub_fig,figsize=(1.5*4, 1*4))
        else:
            fig, axes = plt.subplots(1,num_sub_fig,figsize=(3.6*4, 1*4))
        
    label_list=[]
    for sub_fig_data_dict in subtitle2data_dict.values():
        for label in sub_fig_data_dict.keys():
            label_list.append(label)
    label_list=list(set(label_list))
    

    label_line_dict={}
    
    for (i,(subtitle,sub_fig_data_dict)) in enumerate(sorted(subtitle2data_dict.items())):
        if(more_3_rows):
            sub_ax:Axes=axes[i//4][i%4]
        else:
            if(num_sub_fig==1):
                sub_ax=axes
            else:
                sub_ax:Axes=axes[i]
        if(enable_strize):
            sub_ax.set_title(strize_exp_identifier(subtitle))
        else:
            sub_ax.set_title(subtitle)
        sub_ax.set_xlabel(xlabel)
        sub_ax.set_ylabel(ylabel,labelpad=1 ,rotation=0,loc='top')

        #sub_ax.grid(True)
        if(ylim_bounds is not None):
            sub_ax.set_ylim(ylim_bounds[0],ylim_bounds[1])
        min_length=np.inf

        for j,label in enumerate(label_list):
            if(label=="HCBO-10"):
                continue
            try:
                data=sub_fig_data_dict[label]
                now_length=data.shape[1]
                print(now_length)
                min_length=min(min_length,now_length)
            except Exception as e:
                print(e)

        for j,label in enumerate(label_list):
            if(label=="HCBO-10"):
                continue
            try:
                data=sub_fig_data_dict[label]
                data=data[:,:]
                print(data.shape)
                draw_one_line(data=data,ax=sub_ax,label=label,color=color_list[j])
            except Exception as e:
                print(e)

        lines,labels=sub_ax.get_legend_handles_labels()
        for line,label in zip(lines,labels):
            if(label in label_line_dict):
                assert(line._color== label_line_dict[label]._color)
            else:
                label_line_dict[label]=line
    final_labels=[]
    final_lines=[]
    for label,line in label_line_dict.items():
        final_labels.append(label)
        final_lines.append(line)
    plt.tight_layout()
    #if(more_3_rows):
    if(subplot_distance_dict is not None):
        fig.subplots_adjust(
            **subplot_distance_dict
        )
    #else:
    #    fig.subplots_adjust(top=0.05,bottom=0.10,left=0.1,right=0.1)
    num_legends=len(sub_fig_data_dict)
    fig.legend(final_lines, final_labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)# loc='lower center')
    #fig.legend(final_lines, final_labels, loc='upper center', shadow=True, ncol=num_legends)
    fig.tight_layout()
    plt.tight_layout()
    #for ax in axes:
    #    ax.set_aspect('equal')
    return fig

colors_list=['red','green','blue','orange']

def draw_ablation_es(ax:Axes,best_dict:Dict[str,np.array],hor_line_val):
    df=pd.DataFrame(best_dict)
    x_label="Random Exploration Set"
    y_label="Optimal"
    df_long = df.melt(var_name=x_label, value_name=y_label)
    sns.barplot(x=x_label, y=y_label, data=df_long,ax=ax, palette="Set3")
    ax.axhline(y=hor_line_val, color='red')
