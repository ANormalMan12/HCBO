
## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns



def update_hull(observational_samples, manipulative_variables):
    ## This function computes the coverage of the observations 
    list_variables = []

    for i in range(len(manipulative_variables)):
      list_variables.append(observational_samples[manipulative_variables[i]])

    stack_variables = np.transpose(np.vstack((list_variables)))
    coverage_obs = scipy.spatial.ConvexHull(stack_variables).volume

    return coverage_obs


def observe(num_observation, complete_dataset = None, initial_num_obs_samples = None):
    observational_samples = complete_dataset[initial_num_obs_samples:(initial_num_obs_samples+num_observation)]
    return observational_samples
    

def compute_coverage_mine(manipulative_variables, dict_ranges):
    '''Reture: coverage_total
    , coverage_total'''
    list_ranges = []
    for i in range(len(manipulative_variables)):
      list_ranges.append(dict_ranges[manipulative_variables[i]])
    # NOTE （[data_len, data_dim]）
    vertices = list(itertools.product(*[list_ranges[i] for i in range(len(manipulative_variables))]))
    coverage_total = scipy.spatial.ConvexHull(vertices).volume
    return coverage_total

def compute_coverage(observational_samples, manipulative_variables, dict_ranges):
    '''Reture: the observation and the whole hull ratio, the convex hull Object of observationData, coverage_total'''
    list_variables = []
    list_ranges = []

    for i in range(len(manipulative_variables)):
      list_variables.append(observational_samples[manipulative_variables[i]])
      list_ranges.append(dict_ranges[manipulative_variables[i]])

    # NOTE （[data_len, data_dim]）
    vertices = list(itertools.product(*[list_ranges[i] for i in range(len(manipulative_variables))]))
    coverage_total = scipy.spatial.ConvexHull(vertices).volume

    stack_variables = np.transpose(np.vstack((list_variables)))
    coverage_obs = scipy.spatial.ConvexHull(stack_variables).volume
    hull_obs = scipy.spatial.ConvexHull(stack_variables)

    alpha_coverage = coverage_obs/coverage_total
    return alpha_coverage, hull_obs, coverage_total

# CDS，exploration_set
def preprocessCDS(interventional_data, exploration_set):
  intervention_CDS = []
  for seti in exploration_set:
    for intervention in interventional_data:
      if intervention[1:1+intervention[0]] == seti:
        intervention_CDS.append(intervention)
  if len(exploration_set) != len(intervention_CDS):
    print('error in def preprocessCDS')
    exit()
  return intervention_CDS    

# NOTE ，intervention
def define_initial_data_CBO(interventional_data, num_interventions, exploration_set, name_index, task):
    '''Input:interventional_dataintervention，XY，X
    #? num_interventions，name_indexX，
    Return:data_x_listdata_y_listnum_interventions*|ES|oracle
      ，Y，Y（X），X'''
    data_list = []
    data_x_list = []
    data_y_list = []
    opt_list = []
    gobal_best_y = -np.inf  #!change 
    gobal_best_intervention_value = None#! Change
    gobal_best_intervention_idx = None
    
    for j in range(len(exploration_set)):  # intervention var set
      # len(exploration_set) == len(interventional_data)，
      data = interventional_data["".join(exploration_set[j])].copy() # NOTE ，intervention，intervention
      
      # 
      if task == 'min':
        idx = np.argmin(data[-1]) # Y
        is_batter = True if data[-1][idx] < gobal_best_y else False
      else:
        idx = np.argmax(data[-1])
        is_batter = True if data[-1][idx] > gobal_best_y else False
      if is_batter:
        gobal_best_y = data[-1][idx]
        gobal_best_intervention_value = data[-2][idx]
        gobal_best_intervention_idx = j
      
      # 
      num_variables = data[0]
      if num_variables == 1:
        data_x = np.asarray(data[(num_variables+1)])
        data_y = np.asarray(data[-1])
      else:  # dataOrderedDictx_listy_list
        data_x = np.asarray(data[(num_variables+1):(num_variables*2)][0])
        data_y = np.asarray(data[-1])


      if len(data_y.shape) == 1:
          data_y = data_y[:,np.newaxis]

      if len(data_x.shape) == 1:
          data_x = data_x[:,np.newaxis]
      
      

      all_data = np.concatenate((data_x, data_y), axis =1)

      ## Need to reset the global seed 
      state = np.random.get_state()
      np.random.seed(name_index) # name_indexshufflerandom？
      np.random.shuffle(all_data) # ，
      np.random.set_state(state)

      # ! 
      # subset_all_data = all_data[:num_interventions] # ! : num_interventionsall_data
      # ! : 
      y_max, y_min = np.max(all_data[:,-1]), np.min(all_data[:,-1])
      threshold = y_min + (y_max - y_min) * 0.8
      selected_idx = []
      for i in range(len(all_data)):
         if all_data[i, -1] <= threshold: selected_idx.append(i)
         if len(selected_idx) == num_interventions: break
      subset_all_data = all_data[selected_idx]

      data_list.append(subset_all_data)
      data_x_list.append(data_list[j][:,:-1])
      data_y_list.append(data_list[j][:,-1][:,np.newaxis])


      if task == 'min':
        opt_list.append(np.min(subset_all_data[:,-1])) 
        var_min = exploration_set[np.where(opt_list == np.min(opt_list))[0][0]]
        opt_y = np.min(opt_list)
        opt_intervention_array = data_list[np.where(opt_list == np.min(opt_list))[0][0]]
      else:
        opt_list.append(np.max(subset_all_data[:,-1])) # Y
        var_min = exploration_set[np.where(opt_list == np.max(opt_list))[0][0]] # intervention setYX
        opt_y = np.max(opt_list) # Y
        opt_intervention_array = data_list[np.where(opt_list == np.max(opt_list))[0][0]] # Xintervention（num_interventions）

    # NOTE var_minX，X，；
        # 3
    best_variable=""
    for i in range(len(var_min)):
      best_variable+=var_min[i]

    # YX
    shape_opt = opt_intervention_array.shape[1] - 1
    if task == 'min':
      best_intervention_value = opt_intervention_array[opt_intervention_array[:,shape_opt] == np.min(opt_intervention_array[:,shape_opt]), :shape_opt][0]
    else:  # intervention，
      best_intervention_value = opt_intervention_array[opt_intervention_array[:,shape_opt] == np.max(opt_intervention_array[:,shape_opt]), :shape_opt][0]

    return data_x_list, data_y_list, best_intervention_value, opt_y, best_variable, gobal_best_intervention_value, gobal_best_y, gobal_best_intervention_idx

def define_initial_data_CBO_coverage(interventional_data, num_interventions, exploration_set, name_index, task):
    '''Input:interventional_dataintervention，XY，X
    #? num_interventions，name_indexX，
    Return:data_x_listdata_y_listnum_interventions*|ES|oracle
      ，Y，Y（X），X'''
    data_list = []
    data_x_list = []
    data_y_list = []
    opt_list = []
    
    for j in range(len(exploration_set)):  # intervention var set
      data = interventional_data[j].copy() # NOTE ，intervention，intervention
      
      # 
      num_variables = data[0]
      if num_variables == 1:
        data_x = np.asarray(data[(num_variables+1)])
        data_y = np.asarray(data[-1])
      else:  # dataOrderedDictx_listy_list
        data_x = np.asarray(data[(num_variables+1):(num_variables*2)][0])
        data_y = np.asarray(data[-1])


      if len(data_y.shape) == 1:
          data_y = data_y[:,np.newaxis]

      if len(data_x.shape) == 1:
          data_x = data_x[:,np.newaxis]
      
      

      all_data = np.concatenate((data_x, data_y), axis =1)

      ## Need to reset the global seed 
      state = np.random.get_state()
      np.random.seed(name_index) # name_indexshufflerandom？
      np.random.shuffle(all_data) # ，
      np.random.set_state(state)

      subset_all_data = all_data[:num_interventions]

      data_list.append(subset_all_data)
      data_x_list.append(data_list[j][:,:-1])
      data_y_list.append(data_list[j][:,-1][:,np.newaxis])


      if task == 'min':
        opt_list.append(np.min(subset_all_data[:,-1])) 
        var_min = exploration_set[np.where(opt_list == np.min(opt_list))[0][0]]
        opt_y = np.min(opt_list)
        opt_intervention_array = data_list[np.where(opt_list == np.min(opt_list))[0][0]]
      else:
        opt_list.append(np.max(subset_all_data[:,-1])) # Y
        var_min = exploration_set[np.where(opt_list == np.max(opt_list))[0][0]] # intervention setYX
        opt_y = np.max(opt_list) # Y
        opt_intervention_array = data_list[np.where(opt_list == np.max(opt_list))[0][0]] # Xintervention（num_interventions）

    # NOTE var_minX，X，；
        # 3
    if len(var_min) ==  5:
      best_variable1 = var_min[0]
      best_variable2 = var_min[1]
      best_variable3 = var_min[2]
      best_variable4 = var_min[3]
      best_variable5 = var_min[4]
      best_variable = best_variable1 + best_variable2 + best_variable3 + best_variable4 + best_variable5

    if len(var_min) ==  4:
      best_variable1 = var_min[0]
      best_variable2 = var_min[1]
      best_variable3 = var_min[2]
      best_variable4 = var_min[3]
      best_variable = best_variable1 + best_variable2 + best_variable3 + best_variable4
      
    if len(var_min) ==  3:
      best_variable1 = var_min[0]
      best_variable2 = var_min[1]
      best_variable3 = var_min[2]
      best_variable = best_variable1 + best_variable2 + best_variable3
    
    if len(var_min) ==  2:
      best_variable1 = var_min[0]
      best_variable2 = var_min[1]
      best_variable = best_variable1 + best_variable2
    
    if len(var_min) ==  1:
      best_variable = var_min[0]

    # YX
    shape_opt = opt_intervention_array.shape[1] - 1
    if task == 'min':
      best_intervention_value = opt_intervention_array[opt_intervention_array[:,shape_opt] == np.min(opt_intervention_array[:,shape_opt]), :shape_opt][0]
    else:  # intervention，
      best_intervention_value = opt_intervention_array[opt_intervention_array[:,shape_opt] == np.max(opt_intervention_array[:,shape_opt]), :shape_opt][0]


    return data_x_list, data_y_list, best_intervention_value, opt_y, best_variable

# NOTE ，BO
def define_initial_random_data_CBO(interventional_data, num_interventions, exploration_set, name_index, task):
    interventional_data = []
    return define_initial_data_CBO(interventional_data, num_interventions, exploration_set, name_index, task)