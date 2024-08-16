## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy

from emukit.core import ParameterSpace, ContinuousParameter

## These are functions that operate on the graph structure and allow to sample from a graph, mutilate a graph or to compute the intervention function

def sample_from_model(model, epsilon = None):  #!
  ## Produces a single sample from a structural equation model.
  if epsilon is None:
    # SEM（）
    # ，modelCBOUi
     epsilon = randn(len(model))  #! epsilonGraph
  sample = {}
  for variable, function in model.items():
    sample[variable] = function(epsilon, **sample)  #! sample，，
  return sample



def intervene(*interventions, model):
    new_model = model.copy()
    def assign(model, variable, value):
        model[variable] = lambda epsilon, **kwargs : value
    for variable, value in interventions[0].items():
        assign(new_model, variable, value)
    return new_model


def compute_target_function(*interventions, model, target_variable, num_samples=1000):
    mutilated_model = intervene(*interventions, model = model)

    import concurrent
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(sample_from_model, mutilated_model) for _ in range(num_samples)]
    samples = [future.result() for future in futures]
    #samples = [sample_from_model(mutilated_model) for _ in range(num_samples)] #!
    samples = pd.DataFrame(samples)
    
    return np.mean(samples['Y']), np.var(samples['Y'])



def intervene_dict(model, **interventions):

    new_model = model.copy()

    def assign(model, variable, value): # new_model，，
        model[variable] = lambda epsilon, **kwargs : value
        
    for variable, value in interventions.items():#
        assign(new_model, variable, value)
  
    return new_model


def Intervention_function(*interventions, model, target_variable, 
                          min_intervention, 
                          max_intervention,repeated_times):
    '''modelSEM，model，intervene_dict'''
    num_samples = repeated_times

    assert len(min_intervention) == len(interventions[0])
    assert len(max_intervention) == len(interventions[0])

    def compute_target_function_fcn(value):  # valuex
        '''simulator，（105，115）'''
        num_interventions = len(interventions[0])  # 
        for i in range(num_interventions): # ，interventionsDictXivalue
            interventions[0][list(interventions[0].keys())[i]] = value[0,i]  

        #intervention，SEM，SEM
        # **interventions[0]Listinterventions[0]／，**（）
        # model，mutilated_modelmodelinterventionsvalue
        mutilated_model = intervene_dict(model, **interventions[0]) 
        
        #!state = np.random.get_state()
        #!np.random.seed(1) 
        #NOTE-------BEGIN OF RANDOM STATE-------
        import concurrent
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(sample_from_model, mutilated_model) for _ in range(num_samples)]
        samples = [future.result() for future in futures]
        #samples = [sample_from_model(mutilated_model) for _ in range(num_samples)] #!
        samples = pd.DataFrame(samples)
        #NOTE-------END OF RANDOM STATE-------
        #!np.random.set_state(state)

        return np.asarray(np.mean(samples['Y']))[np.newaxis,np.newaxis]
    
    ## Define parameter space
    list_parameter = [None]*len(interventions[0])
    for i in range(len(interventions[0])):
        list_parameter[i] = ContinuousParameter(list(interventions[0].keys())[i], 
                                                min_intervention[i], max_intervention[i])
        
    return (compute_target_function_fcn, ParameterSpace(list_parameter))
