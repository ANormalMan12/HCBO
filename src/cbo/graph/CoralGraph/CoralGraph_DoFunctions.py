import numpy as np
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

def fit_single_GP_model(X, Y, parameter_list, ard = False):
    kernel = RBF(X.shape[1], ARD = parameter_list[3], lengthscale=parameter_list[0], variance = parameter_list[1]) 
    gp = GPRegression(X = X, Y = Y, kernel = kernel, noise_var= parameter_list[2])
    gp.likelihood.variance.fix(1e-2)
    gp.optimize()
    return gp
################################ One variable Do function
def compute_do_N(observational_samples, functions, value):
    gp_N = functions['gp_N']

    mean_do = np.mean(gp_N.predict(np.ones((1,1))*value)[0])
    var_do = np.mean(gp_N.predict(np.ones((1,1))*value)[1])

    return mean_do, var_do


def compute_do_O(observational_samples, functions, value):

    gp_O_S_T_D_T = functions['gp_O_S_T_D_E']
    
    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    T = np.asarray(observational_samples['T'])[:,np.newaxis]
    D = np.asarray(observational_samples['D'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value, S.shape[0])[:,np.newaxis], S, T, D, E))

    mean_do = np.mean(gp_O_S_T_D_T.predict(intervened_inputs)[0])
    
    var_do = np.mean(gp_O_S_T_D_T.predict(intervened_inputs)[1])

    return mean_do, var_do
   


def compute_do_C(observational_samples, functions, value):
    
    gp_C_N_L_T = functions['gp_C_N_L_E']
    
    N = np.asarray(observational_samples['N'])[:,np.newaxis]
    L = np.asarray(observational_samples['L'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value, N.shape[0])[:,np.newaxis], N, L, E))
    
    mean_do = np.mean(gp_C_N_L_T.predict(intervened_inputs)[0])
    
    var_do = np.mean(gp_C_N_L_T.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_T(observational_samples, functions, value):
    
    gp_T_S = functions['gp_T_S']
    
    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value, S.shape[0])[:,np.newaxis],S))
    
    mean_do = np.mean(gp_T_S.predict(intervened_inputs)[0])
    
    var_do = np.mean(gp_T_S.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_D(observational_samples, functions, value):
    
    gp_D_S = functions['gp_D_S']
    
    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value, S.shape[0])[:,np.newaxis],S))
    
    mean_do = np.mean(gp_D_S.predict(intervened_inputs)[0])
    
    var_do = np.mean(gp_D_S.predict(intervened_inputs)[1])

    return mean_do, var_do


################################################## Two variables Do function


def compute_do_NO(observational_samples, functions, value):

    gp_N_O_S_T_D_T = functions['gp_N_O_S_T_D_E']

    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    T = np.asarray(observational_samples['T'])[:,np.newaxis]
    D = np.asarray(observational_samples['D'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   S, T, D, E))
    mean_do = np.mean(gp_N_O_S_T_D_T.predict(intervened_inputs)[0])
    
    var_do = np.mean(gp_N_O_S_T_D_T.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_NC(observational_samples, functions, value):

    gp_C_N_L_T = functions['gp_C_N_L_E']

    L = np.asarray(observational_samples['L'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    
    
    intervened_inputs = np.hstack((np.repeat(value[1], L.shape[0])[:,np.newaxis], 
                                   np.repeat(value[0], L.shape[0])[:,np.newaxis], 
                                   L, E))
    mean_do = np.mean(gp_C_N_L_T.predict(intervened_inputs)[0])
    
    var_do = np.mean(gp_C_N_L_T.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_NT(observational_samples, functions, value):

    gp_N_T_S = functions['gp_N_T_S']

    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   S))
    mean_do = np.mean(gp_N_T_S.predict(intervened_inputs)[0])
    
    var_do = np.mean(gp_N_T_S.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_ND(observational_samples, functions, value):

    gp_N_D_S = functions['gp_N_D_S']

    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   S))
    mean_do = np.mean(gp_N_D_S.predict(intervened_inputs)[0])
    
    var_do = np.mean(gp_N_D_S.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_OC(observational_samples, functions, value):

    gp_O_C_N_L_E_S_T_D = functions['gp_O_C_N_L_E_S_T_D']

    N = np.asarray(observational_samples['N'])[:,np.newaxis]
    L = np.asarray(observational_samples['L'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    T = np.asarray(observational_samples['T'])[:,np.newaxis]
    D = np.asarray(observational_samples['D'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   N,L,E,S,T,D))
    mean_do = np.mean(gp_O_C_N_L_E_S_T_D.predict(intervened_inputs)[0])
    
    var_do = np.mean(gp_O_C_N_L_E_S_T_D.predict(intervened_inputs)[1])

    return mean_do, var_do



def compute_do_OC(observational_samples, functions, value):

    gp_O_C_N_L_E_S_T_D = functions['gp_O_C_N_L_E_S_T_D']

    N = np.asarray(observational_samples['N'])[:,np.newaxis]
    L = np.asarray(observational_samples['L'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    T = np.asarray(observational_samples['T'])[:,np.newaxis]
    D = np.asarray(observational_samples['D'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   N,L,E,S,T,D))
    mean_do = np.mean(gp_O_C_N_L_E_S_T_D.predict(intervened_inputs)[0])
    
    var_do = np.mean(gp_O_C_N_L_E_S_T_D.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_OT(observational_samples, functions, value):

    gp_O_S_T_D_T = functions['gp_O_S_T_D_E']

    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    D = np.asarray(observational_samples['D'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], S,
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   D,E))
    mean_do = np.mean(gp_O_S_T_D_T.predict(intervened_inputs)[0])
    
    var_do = np.mean(gp_O_S_T_D_T.predict(intervened_inputs)[1])

    return mean_do, var_do



def compute_do_OD(observational_samples, functions, value):

    gp_O_S_T_D_T = functions['gp_O_S_T_D_E']

    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    T = np.asarray(observational_samples['T'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], S, T,
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   E))
    mean_do = np.mean(gp_O_S_T_D_T.predict(intervened_inputs)[0])

    var_do = np.mean(gp_O_S_T_D_T.predict(intervened_inputs)[1])

    return mean_do, var_do



def compute_do_TC(observational_samples, functions, value):

    gp_T_C_S_E_L_N = functions['gp_T_C_S_E_L_N']

    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    N = np.asarray(observational_samples['N'])[:,np.newaxis]
    L = np.asarray(observational_samples['L'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   S,E,L,N))
    mean_do = np.mean(gp_T_C_S_E_L_N.predict(intervened_inputs)[0])

    var_do = np.mean(gp_T_C_S_E_L_N.predict(intervened_inputs)[1])

    return mean_do, var_do



def compute_do_TD(observational_samples, functions, value):

    gp_T_D_S = functions['gp_T_D_S']

    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   S))
    mean_do = np.mean(gp_T_D_S.predict(intervened_inputs)[0])

    var_do = np.mean(gp_T_D_S.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_CD(observational_samples, functions, value):

    gp_C_D_S_E_L_N = functions['gp_C_D_S_E_L_N']

    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    L = np.asarray(observational_samples['L'])[:,np.newaxis]
    N = np.asarray(observational_samples['N'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   S, E, L, N))
    mean_do = np.mean(gp_C_D_S_E_L_N.predict(intervened_inputs)[0])

    var_do = np.mean(gp_C_D_S_E_L_N.predict(intervened_inputs)[1])

    return mean_do, var_do


################################################## Three variables Do function

def compute_do_NOC(observational_samples, functions, value):

    gp_O_C_N_L_E_S_T_D = functions['gp_O_C_N_L_E_S_T_D']

    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    L = np.asarray(observational_samples['L'])[:,np.newaxis]
    T = np.asarray(observational_samples['T'])[:,np.newaxis]
    D = np.asarray(observational_samples['D'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[2], S.shape[0])[:,np.newaxis],
                                   np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   L,E,S,T,D))
    mean_do = np.mean(gp_O_C_N_L_E_S_T_D.predict(intervened_inputs)[0])

    var_do = np.mean(gp_O_C_N_L_E_S_T_D.predict(intervened_inputs)[1])

    return mean_do, var_do



def compute_do_NOT(observational_samples, functions, value):

    gp_N_O_S_T_D_T = functions['gp_N_O_S_T_D_E']

    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    D = np.asarray(observational_samples['D'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis],S,
                                   np.repeat(value[2], S.shape[0])[:,np.newaxis], 
                                   D,E))
    mean_do = np.mean(gp_N_O_S_T_D_T.predict(intervened_inputs)[0])

    var_do = np.mean(gp_N_O_S_T_D_T.predict(intervened_inputs)[1])

    return mean_do, var_do



def compute_do_NOD(observational_samples, functions, value):

    gp_N_O_S_T_D_T = functions['gp_N_O_S_T_D_E']

    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    T = np.asarray(observational_samples['T'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis],S,T,
                                   np.repeat(value[2], S.shape[0])[:,np.newaxis], 
                                   E))
    mean_do = np.mean(gp_N_O_S_T_D_T.predict(intervened_inputs)[0])

    var_do = np.mean(gp_N_O_S_T_D_T.predict(intervened_inputs)[1])

    return mean_do, var_do



def compute_do_NCT(observational_samples, functions, value):

    gp_N_C_T_S_N_L_T = functions['gp_N_C_T_S_N_L_E']

    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    N = np.asarray(observational_samples['N'])[:,np.newaxis]
    L = np.asarray(observational_samples['L'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis],
                                   np.repeat(value[2], S.shape[0])[:,np.newaxis], 
                                   S,N,L,E))
    mean_do = np.mean(gp_N_C_T_S_N_L_T.predict(intervened_inputs)[0])

    var_do = np.mean(gp_N_C_T_S_N_L_T.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_NCD(observational_samples, functions, value):

    gp_C_D_S_E_L_N = functions['gp_C_D_S_E_L_N']

    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    L = np.asarray(observational_samples['L'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[2], S.shape[0])[:,np.newaxis],S,E,L,
                                   np.repeat(value[0], S.shape[0])[:,np.newaxis]))
    mean_do = np.mean(gp_C_D_S_E_L_N.predict(intervened_inputs)[0])

    var_do = np.mean(gp_C_D_S_E_L_N.predict(intervened_inputs)[1])

    return mean_do, var_do




def compute_do_NTD(observational_samples, functions, value):

    gp_N_T_D_S = functions['gp_N_T_D_S']

    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis],
                                   np.repeat(value[2], S.shape[0])[:,np.newaxis],S))
    mean_do = np.mean(gp_N_T_D_S.predict(intervened_inputs)[0])

    var_do = np.mean(gp_N_T_D_S.predict(intervened_inputs)[1])

    return mean_do, var_do



def compute_do_OCT(observational_samples, functions, value):

    gp_O_C_N_L_E_S_T_D = functions['gp_O_C_N_L_E_S_T_D']

    N = np.asarray(observational_samples['N'])[:,np.newaxis]
    L = np.asarray(observational_samples['L'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    D = np.asarray(observational_samples['D'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], N, L, E, S, 
                                   np.repeat(value[2], S.shape[0])[:,np.newaxis],D))
    mean_do = np.mean(gp_O_C_N_L_E_S_T_D.predict(intervened_inputs)[0])

    var_do = np.mean(gp_O_C_N_L_E_S_T_D.predict(intervened_inputs)[1])

    return mean_do, var_do





def compute_do_OCT(observational_samples, functions, value):

    gp_O_C_N_L_E_S_T_D = functions['gp_O_C_N_L_E_S_T_D']

    N = np.asarray(observational_samples['N'])[:,np.newaxis]
    L = np.asarray(observational_samples['L'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    D = np.asarray(observational_samples['D'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], N, L, E, S, 
                                   np.repeat(value[2], S.shape[0])[:,np.newaxis],D))
    mean_do = np.mean(gp_O_C_N_L_E_S_T_D.predict(intervened_inputs)[0])

    var_do = np.mean(gp_O_C_N_L_E_S_T_D.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_OCD(observational_samples, functions, value):

    gp_O_C_N_L_E_S_T_D = functions['gp_O_C_N_L_E_S_T_D']

    N = np.asarray(observational_samples['N'])[:,np.newaxis]
    L = np.asarray(observational_samples['L'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    T = np.asarray(observational_samples['T'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], N, L, E, S, T,
                                   np.repeat(value[2], S.shape[0])[:,np.newaxis]))
    mean_do = np.mean(gp_O_C_N_L_E_S_T_D.predict(intervened_inputs)[0])

    var_do = np.mean(gp_O_C_N_L_E_S_T_D.predict(intervened_inputs)[1])

    return mean_do, var_do




def compute_do_CTD(observational_samples, functions, value):

    gp_C_T_D_S_N_L_T = functions['gp_C_T_D_S_N_L_E']

    L = np.asarray(observational_samples['L'])[:,np.newaxis]
    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    N = np.asarray(observational_samples['N'])[:,np.newaxis]
    

    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[2], S.shape[0])[:,np.newaxis], S, N, L, E))
    mean_do = np.mean(gp_C_T_D_S_N_L_T.predict(intervened_inputs)[0])

    var_do = np.mean(gp_C_T_D_S_N_L_T.predict(intervened_inputs)[1])

    return mean_do, var_do





def compute_do_OTD(observational_samples, functions, value):

    gp_O_S_T_D_T = functions['gp_O_S_T_D_E']

    E = np.asarray(observational_samples['E'])[:,np.newaxis]
    S = np.asarray(observational_samples['S'])[:,np.newaxis]
    

    intervened_inputs = np.hstack((np.repeat(value[0], S.shape[0])[:,np.newaxis], S,
                                   np.repeat(value[1], S.shape[0])[:,np.newaxis], 
                                   np.repeat(value[2], S.shape[0])[:,np.newaxis], E))
    mean_do = np.mean(gp_O_S_T_D_T.predict(intervened_inputs)[0])

    var_do = np.mean(gp_O_S_T_D_T.predict(intervened_inputs)[1])

    return mean_do, var_do




