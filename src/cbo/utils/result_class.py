import pandas as pd
import pathlib
import pickle
from tabulate import tabulate
class result_class():
	def __init__(self,run_cfg,seed,exploration_set,task):
		self.record_dict=run_cfg
		self.record_dict['seed']=seed
		self.record_dict['task']=task
		self.step_data=pd.DataFrame(
			columns=['step','history','cost','type','update_index','all_acqusition_y']
		)
		self.exploration_set_list=pd.DataFrame(exploration_set)

	def add_step_result(self,history,cost,type,update_index,all_acqusition_y,):
		self.step_data=self.step_data.append(
			{
				'step':len(self.step_data),
				'history':history,
				'cost':cost,
				'type':type,
				'update_index':update_index,
				'all_acqusition_y':all_acqusition_y,
			}
			,ignore_index=True
		)
	def get_last_step_type(self):
		return self.step_data.iloc[-1]['type']
	def final_add(self,current_best_x, current_best_y,total_time,date_string,data_cfg_name):
		self.data_cfg_name=data_cfg_name
		self.current_best=pd.DataFrame({
			"current_best_x":current_best_x,
			"current_best_y":current_best_y
			}
		)
		self.record_dict['total_time']=total_time
		self.record_dict['date_string']=date_string
		if(self.record_dict['task']=='max'):
			self.record_dict['global_opt']=self.step_data['history'].max()
		else:
			self.record_dict['global_opt']=self.step_data['history'].min()
	def get_acquisition_df(self):
		subset_data=pd.DataFrame(
			columns=['set','acqusition_per_step']
		)
		num_sets=len(self.exploration_set_list)
		for i in range(num_sets):
			subset_data=subset_data.append(
				{
					'set':self.to_str_exploration_set_item(self.exploration_set_list.iloc[i]),
					'acqusition_per_step':self.step_data['all_acqusition_y'].apply(
						lambda x:(x[i][0][0] if x[i]!=None else None)
					).tolist()
				}
				,ignore_index=True
			)
		return subset_data
	def get_opt_lists(self):
		
		history_index_columns=self.step_data[['history','update_index']]
		#print(history_index_columns.head())
		opt_list=[]
		optAvg_list=[]
		
		#print(history_index_columns)
		nowOpt=nowOptAvg=None
		for i,data in history_index_columns.iterrows():
			if(i==0):continue
			history,index=data
			if(index!=-1):
				nowOpt=history
				dim=self.exploration_set_list.iloc[index].count()
				#print(dim)
				nowOptAvg=history/dim

			if(opt_list!=[]):
				if(self.record_dict['task']=='max'):
					nowOpt   =max(opt_list[-1],nowOpt)
					nowOptAvg=max(optAvg_list[-1],nowOptAvg)
				elif(self.record_dict['task']=='min'):
					nowOpt   =min(opt_list[-1],nowOpt)
					nowOptAvg=min(optAvg_list[-1],nowOptAvg)
				else:
					raise("Failed task")
			opt_list.append(nowOpt)
			optAvg_list.append(nowOptAvg)
			
		return opt_list,optAvg_list
	
	def get_exploration_set_update_times(self):
		update_index_list=self.step_data['update_index'].tolist()
		exploration_set_update_times=pd.DataFrame()
		for i in range(len(self.exploration_set_list)):
			exploration_set_update_times=exploration_set_update_times.append(
				[{
					'set':self.to_str_exploration_set_item(self.exploration_set_list.iloc[i]),
					'times':update_index_list.count(i)
				}]
			)
		return exploration_set_update_times
	
	def to_str_exploration_set_item(self,exploration_set_item)->str:
		ret=""
		for s in exploration_set_item:
			if(s!=None):
				ret+=s
		return ret
	
	def test(self):
		for j in range(len(self.exploration_set_list)):
			print(j,"------Begin------")
			print(self.exploration_set_list.iloc[j])
			print('--------str----------')
			print(self.to_str_exploration_set_item(self.exploration_set_list.iloc[j]))
			print(j,"------End------")
	def save(self,folder):
		path='./Results/'+folder+'/'
		print(path)
		pathlib.Path(path).mkdir(parents=True, exist_ok=False)
		pklPath=path+"result.pkl"
		with open(pklPath,'wb') as f:
			pickle.dump(self,f)
		return pklPath


def show_pd_table(dataf):
    table = tabulate(dataf,headers='keys',tablefmt='fancy_grid')
    print(table)

def get_opt_data(selected_libs):
	opt_data=pd.DataFrame(columns=['step','opt','opt_avg'])
	for folder in selected_libs['artifact_uri']:
		path=folder+'/'+'result.pkl'
		try:
			with open(path, "rb") as file:
				result:result_class = pickle.load(file)
		except:
			continue    
		show_pd_table(result.step_data)
		opt_list,opt_avg_list=result.get_opt_lists()
		for i in range(len(opt_list)):
			opt_data=opt_data.append({'step':i,'opt':opt_list[i],'opt_avg':opt_avg_list[i]},ignore_index=True)
	return opt_data
def get_acquisition_data(selected_libs):
    acq_df=pd.DataFrame()
    for folder in selected_libs['artifact_uri']:
        path=folder+'/'+'result.pkl'
        with open(path, "rb") as file:
            result:result_class = pickle.load(file)
        acq_df=acq_df.append(result.get_acquisition_df())
    return acq_df
def get_update_times(selected_libs):
    up_df=pd.DataFrame()
    for folder in selected_libs['artifact_uri']:
        path=folder+'/'+'result.pkl'
        with open(path, "rb") as file:
            result:result_class = pickle.load(file)
        up_df=up_df.append(result.get_exploration_set_update_times())
    return up_df