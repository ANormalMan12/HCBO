import numpy as np
from ..utils import *
from .utils import *
from ..sem.sem_synt import SEM_synt
from .graph import CoralGraph
def get_interv_set_from_CBO_mode(IS,name_num):
	interv_set=[]
	for c in IS:
		interv_set.append(name_num[c])
	return tuple(sorted(interv_set))

def fixed_cost_function(**kwargs):
	return 1

def get_range_lists(target_sem:SEM_synt,num_name):
	range_lists=[]
	for i in range(target_sem.get_connection().shape[0]):
		now_bounds=target_sem.get_bounds((i,))
		range_lists.append(
			(num_name[i],[
				now_bounds[0][0],
				now_bounds[1][0]
			])
		)
	return range_lists

def optimize_with_CBO(
	target_sem:SEM_synt,
	intervention_data_dict,
	full_observational_samples,
	optimization_cost,
	es_hcbo_form,
	Causal_prior:bool
	):
	if(Causal_prior):
		graph=CoralGraph(full_observational_samples,full_observational_samples)
	num_name=target_sem.num_name
	initial_num_obs_samples     = len(full_observational_samples)//2
	num_additional_observations = min(10,len(full_observational_samples)//2)
	observational_samples = full_observational_samples[:initial_num_obs_samples]
	max_N = len(full_observational_samples)
	cost_avg = False
	task=target_sem.task
	manipulative_variables=[num_name[index] for index in target_sem.get_intervenable_variables()]
	exploration_set=[
		[num_name[index] for index in tup]for tup in es_hcbo_form
	]
	print("index to name:",num_name)
	print("Exploration Set(index):",es_hcbo_form)
	print("Exploration Set(name):",exploration_set)
	bounds=target_sem.get_bounds(target_sem.get_intervenable_variables())
	dict_ranges = OrderedDict([
		(manipulative_variables[i],[bounds[0][i],bounds[1][i]])
		for i in range(bounds.shape[1])
	])

	costs=OrderedDict([
		(num_name[i],fixed_cost_function)
		for i in range(len(manipulative_variables))
	])

	alpha_coverage, hull_obs, coverage_total = compute_coverage(observational_samples, manipulative_variables, dict_ranges)

	current_best_x, current_best_y, x_dict_mean, x_dict_var, dict_interventions = initialise_dicts(exploration_set, target_sem.task)

	space_list=[]
	for index_is,IS in enumerate(exploration_set):
		list_parameter = []
		for i in range(len(IS)):
			list_parameter.append(ContinuousParameter(
				IS[i],
				dict_ranges[IS[i]][0],dict_ranges[IS[i]][1])   
			)
		space_list.append(ParameterSpace(list_parameter))
	model_list = [None]*len(exploration_set)
	enable_list= [True]*len(exploration_set)
	current_global=-np.inf if target_sem.task=='max' else np.inf
	data_x_list, data_y_list=[],[]
	
	result_x_list=[]
	result_y_list=[]
	visit_IS_list=[]

	accumulated_cost=0
	for index,intervention_set in enumerate(exploration_set):
		data_x_y=intervention_data_dict[es_hcbo_form[index]]
		accumulated_cost+=len(intervention_set)*len(data_x_y[0])
		now_xs=np.array(data_x_y[0])
		now_ys=np.array(data_x_y[1])
		data_x_list.append(now_xs)
		data_y_list.append(now_ys)
		for i,x in enumerate(now_xs):
			result_x_list.append(x)
			result_y_list.append(now_ys[i][0])
			print(now_ys[i][0])
			visit_IS_list.append(intervention_set)
		if(target_sem.task=='max'):
			current_global=max(current_global,np.max(data_y_list[-1]))
		else:
			current_global=min(current_global,np.min(data_y_list[-1]))
	mean_functions_list=[None]*len(exploration_set)
	var_functions_list=[None]*len(exploration_set)
	last_type="o"
	
	print("Init Cost for CBO:",accumulated_cost)
	i=-1
	while(True):
		print(accumulated_cost)
		if(accumulated_cost>=optimization_cost):
			break  
		i+=1
		coverage_obs = update_hull(observational_samples, manipulative_variables)
		rescale = observational_samples.shape[0]/max_N 
		epsilon_coverage = (coverage_obs/coverage_total)/rescale
		uniform = np.random.uniform(0.,1.)
		
		if i == 0:
			uniform = 0.
		if(not Causal_prior):
			uniform=np.inf
		if uniform < epsilon_coverage:
			new_observational_samples = observe(num_observation = num_additional_observations, 
												complete_dataset = full_observational_samples, 
												initial_num_obs_samples= initial_num_obs_samples)
			observational_samples = observational_samples.append(new_observational_samples)
			functions = graph.refit_models(observational_samples)
			mean_functions_list, var_functions_list = update_all_do_functions(graph, exploration_set, functions, dict_interventions, 
														observational_samples, x_dict_mean, x_dict_var)
			last_type='o'
		else: 
			y_acquisition_list = [None]*len(exploration_set)
			x_new_list = [None]*len(exploration_set)

			if last_type=='o':# When the last time is observation
				for s in range(len(exploration_set)):
					print('Updating model:', s)
					model_list[s] = update_BO_models(mean_functions_list[s], var_functions_list[s], data_x_list[s], data_y_list[s], Causal_prior)
			else: 
				print('Updating model:', index)
				model_list[index] = update_BO_models(
					mean_functions_list[index],
					var_functions_list[index],
					data_x_list[index], data_y_list[index]
					,Causal_prior,
				)
				

			## Compute acquisition function given the updated BO models for the interventional data
			##? Notice that we use current_global and the costs to compute the acquisition functions 
				
			for s in range(len(exploration_set)):
				if(not enable_list[s]):
					y_acquisition_list[s]=np.array([[-np.inf]])
					x_new_list[s]=None
				else:	
					y_acquisition_list[s], x_new_list[s] = find_next_y_point(space_list[s], model_list[s], current_global, exploration_set[s], costs, task = task, cost_avg=cost_avg)
				

			## Selecting the variable to intervene based on the values of the acquisition functions
			#NOTE change on var
			
			index=np.argmax(y_acquisition_list)
			var_to_intervene = exploration_set[index]
			def target_function(X,target_is):
				interv_plan=[]
				for i,x in enumerate(X):
					interv_plan.append((target_is[i],x))
				Y=target_sem.intervene(1,SEM_REPEATED_TIMES,interv_plan)
				return Y
			y_new = [target_function(x_new_list[index][0],es_hcbo_form[index])]
			y_val=y_new[0][0]
			if(target_sem.task=='max'):
				current_global=max(current_global,y_val)
			else:
				current_global=min(current_global,y_val)
			print('Selected intervention: ', var_to_intervene)
			print('Selected point: ', x_new_list[index])
			print('Target function at selected point: ', y_new)

			data_x, data_y_x = add_data([data_x_list[index], data_y_list[index]], 
													[x_new_list[index], y_new])
			data_x_list[index] = np.vstack((data_x_list[index], x_new_list[index])) 
			data_y_list[index] = np.vstack((data_y_list[index], y_new))
			try:
				model_list[index].set_data(data_x, data_y_x)  # NOTE set_dataemukit，GP data
				model_list[index].optimize()
			except:
				enable_list[index]=False
				past_index=index
				while(index==past_index):
					index=random.randint(0,len(es_hcbo_form)-1)
				continue
			#x_new_dict = get_new_dict_x(x_new_list[index], dict_interventions[index])  # ，x_new_dict 
			last_type='i'
		result_x_list.append(x_new_list[index][0])
		result_y_list.append(y_val)
		visit_IS_list.append(exploration_set[index])
		accumulated_cost+=len(x_new_list[index][0])
	return result_x_list,result_y_list,visit_IS_list,{
            "Set_Selector_Type":"CBO",
            "acq_history":[],
            "mean_history":[],
            "issf_history":[]
        }