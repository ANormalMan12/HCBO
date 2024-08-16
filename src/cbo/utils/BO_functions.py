## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns

import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper


from .causal_kernels import CausalRBF


import os
# import time
# import tool

def define_initial_data_BO(interventional_data, num_interventions, intervention_sets, name_index, ):
    data_x = (interventional_data[0][len(intervention_sets)]).copy()
    data_y = (np.asarray(interventional_data[0][len(intervention_sets)+1])).copy()
    all_data = np.concatenate((data_x, data_y), axis =1)

    ## Need to reset the global seed 
    state = np.random.get_state()

    np.random.seed(name_index)
    np.random.shuffle(all_data)
    
    np.random.set_state(state)

    data_x = all_data[:num_interventions, :len(intervention_sets)]
    data_y = all_data[:num_interventions, len(intervention_sets):]


    data_list = [all_data]

    min_y = np.min(data_y)
    min_intervention_value = np.transpose(all_data[np.where(data_y == min_y)[0][0]][:len(intervention_sets)][:,np.newaxis])

    return data_x, data_y, min_intervention_value, min_y

'''GP
lengthscale, Default: No lengthscale (i.e. lengthscale is the identity matrix).
'''

def update_BO_models(mean_function, var_function, data_x, data_y, Causal_prior):    
    ## This function updates the BO model for each intervetion set 
    if Causal_prior==False:
        # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! RBF 
        #                                       GPy.kern.RBF(data_x.shape[1], lengthscale=1., variance=1.), 
        #                                           noise_var=1e-10)
        # gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! Linear 12
        #                                       GPy.kern.Linear(data_x.shape[1]), 
        #                                           noise_var=1e-10)
        gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! Matern52 
                                              GPy.kern.Matern52(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! Matern32 
        #                                       GPy.kern.Matern32(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! No implemented gradients_X
        #                                       GPy.kern.PeriodicExponential(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! No implemented Periodic kernels are only defined for input_dim=1
        #                                       GPy.kern.PeriodicMatern32())
        # gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! No implemented
        #                                       GPy.kern.PeriodicMatern52(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! No implemented
        #                                       GPy.kern.PolynomialBasisFuncKernel(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y,   # ! No implemented
        #                                       GPy.kern.Poly(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y,   # ! RatQuad (RQKernal) 
        #                                       GPy.kern.RatQuad(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! BasisFuncKernel, 
        #                                       GPy.kern.ChangePointBasisFuncKernel(data_x.shape[1],np.mean(data_x, axis=0)))
        # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! DiffKern Kernel
        #                                       GPy.kern.DiffKern(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! ExpQuad 
        #                                       GPy.kern.ExpQuad(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! GridRBF(RBF) 
        #                                       GPy.kern.GridRBF(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! Hierarchical Kernel
        #                                       GPy.kern.Hierarchical(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! TruncLinear_inf interval, , .
        #                                       GPy.kern.TruncLinear_inf(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! TruncLinear 5.
        #                                       GPy.kern.TruncLinear(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! , transform
        #                                       GPy.kern.Symmetric(data_x.shape[1]))
        # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! StdPeriodic, 8, .
        #                                       GPy.kern.StdPeriodic(data_x.shape[1]))

    else:    
        mf = GPy.core.Mapping(data_x.shape[1], 1)
        mf.f = lambda x: mean_function(x)
        mf.update_gradients = lambda a, b: None
        causal_kernel = CausalRBF(data_x.shape[1], variance_adjustment=var_function, 
                                            lengthscale=1., variance=1., ARD = False)

        # GPy.models.GPRegression，（kernel&mean）(data)，
        gpy_model = GPy.models.GPRegression(data_x, data_y, causal_kernel, 
                                                      noise_var=1e-10, mean_function=mf)
    
    #gpy_model.likelihood.variance.fix(1e-2) 
    model = GPyModelWrapper(gpy_model)
    model.optimize()
    # DEBUG
    # if plot:
    #     fig1 = gpy_model.plot()
    #     fname = os.path.join('figture_CoralGraphCausalFalse_demo', tool.getTime())
    #     # fig1.savefig(fname=fname)
        
    
    return model

def update_BO_models_mine(mean_function, var_function, data_x, data_y, Causal_prior, plot):    
    ## This function updates the BO model for each intervetion set 
    gpy_model = GPy.models.GPRegression(data_x, data_y, # ! RBF 
                                          GPy.kern.RBF(data_x.shape[1], lengthscale=1., variance=1.), 
                                              noise_var=1e-10)
    # gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! Linear 12
    #                                       GPy.kern.Linear(data_x.shape[1]), 
    #                                           noise_var=1e-10)
    # gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! Matern52 
    #                                         GPy.kern.Matern52(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! Matern32 
    #                                       GPy.kern.Matern32(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! No implemented gradients_X
    #                                       GPy.kern.PeriodicExponential(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! No implemented Periodic kernels are only defined for input_dim=1
    #                                       GPy.kern.PeriodicMatern32())
    # gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! No implemented
    #                                       GPy.kern.PeriodicMatern52(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y,  # ! No implemented
    #                                       GPy.kern.PolynomialBasisFuncKernel(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y,   # ! No implemented
    #                                       GPy.kern.Poly(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y,   # ! RatQuad (RQKernal) 
    #                                       GPy.kern.RatQuad(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! BasisFuncKernel, 
    #                                       GPy.kern.ChangePointBasisFuncKernel(data_x.shape[1],np.mean(data_x, axis=0)))
    # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! DiffKern Kernel
    #                                       GPy.kern.DiffKern(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! ExpQuad 
    #                                       GPy.kern.ExpQuad(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! GridRBF(RBF) 
    #                                       GPy.kern.GridRBF(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! Hierarchical Kernel
    #                                       GPy.kern.Hierarchical(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! TruncLinear_inf interval, , .
    #                                       GPy.kern.TruncLinear_inf(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! TruncLinear 5.
    #                                       GPy.kern.TruncLinear(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! , transform
    #                                       GPy.kern.Symmetric(data_x.shape[1]))
    # gpy_model = GPy.models.GPRegression(data_x, data_y, # ! StdPeriodic, 8, .
    #                                       GPy.kern.StdPeriodic(data_x.shape[1]))
    
    model = GPyModelWrapper(gpy_model) # ! , model_list[index].set_data
    
    model.optimize()
        
    
    return model
