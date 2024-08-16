class Weight_Difference():
    def plot_weight(weight_dict_sample,weight_dict_oracle,ax,label):
        x=[]
        y=[]
        for key,val in weight_dict_sample.items():
            x.append(val)
            y.append(weight_dict_oracle[key])
        sns.regplot(x=x,y=y,ax=ax,label=label, scatter_kws={"alpha": 0.5})

    def transport_oracle(oracle_data,task,length:int):
        oracle_Ys=[None]*len(oracle_data)
        for i in range(len(oracle_data)):
            oracle_Ys[i]=oracle_data[i][1]
        Y_star_dict={}
        for i in range(len(oracle_Ys)):
            if(task=="max"):
                Y_star_dict[i]=oracle_Ys[i][:length].max().item()
            else:
                Y_star_dict[i]=oracle_Ys[i][:length].min().item()
        now_weight=get_weight_dict(Ystar_dict=Y_star_dict)
        return now_weight

    def get_compare_weight(use_weight,oracle_weight):
        fig,ax=plt.subplots()
        ax.set_xlabel('Calculated Weight')
        ax.set_ylabel('Oracle Weight')
        
        plot_weight(use_weight,oracle_weight,ax=ax,label="use_weight")
        plot_weight(oracle_weight,oracle_weight,ax=ax,label="oracle_weight")
        
        ax.legend()
        plt.tight_layout()
        current_ylim = ax.get_ylim()
        ax.set_ylim(0, current_ylim[1])
        current_xlim = ax.get_xlim()
        ax.set_xlim(0, current_xlim[1])

        return fig

    def get_save_weight(sem_name,the_sem):
        weight,data=get_weight_and_data_x_y_var_dict(the_sem,repeated_times,7)
        save_json(weight,weight_folder/f"weight-{sem_name}-{repeated_times}.json")
        save_pickle(data,weight_folder/f"sample-{sem_name}-{repeated_times}.pkl")
        return weight,data
    def main():
        import argparse
        parser = argparse.ArgumentParser(description="sem Analysis")
        parser.add_argument("exp_name",type=str)
        repeated_times=20
        args=parser.parse_args()
        exp_folder:pathlib.Path=GRAPH_DIR_PATH/args.exp_name
        weight_folder:pathlib.Path=exp_folder/"weight"
        oracle_sem:SEM_synt=read_pickle(exp_folder/"oracle_sem.pkl")
        fitted_sem:SEM_synt=read_pickle(exp_folder/"1000-fitted_sem.pkl")

        (oracle_weight,oracle_data)=get_save_weight("oracle",oracle_sem)
        (fitted_weight,fitted_data)=get_save_weight("fitted",fitted_sem)
        
        fig=get_compare_weight(oracle_weight=oracle_weight,use_weight=fitted_weight)
        fig.save_fig(weight_folder/"oracle_fitted_weight_compare.png")