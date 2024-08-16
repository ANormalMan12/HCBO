
# from asyncio import futures
# from concurrent.futures import process
# import multiprocessing
# from pathos.pools import ProcessPool as Pool
# import pathos

# ## My functions
# from copyreg import pickle
# from json import tool
# from multiprocessing import pool
# from multiprocessing import popen_fork
# from re import S
# from unittest import result
# from utils_functions import *
from typing import List
from draftCommonExperiment import *
# from graphs import * 

import tool.algorithm
from tqdm import tqdm
# # from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib


def cal_intervention(num_trials, exploration_set, manipulative_variables, data_x_list, data_y_list,  best_intervention_value, opt_y, 
					best_variable, dict_ranges, functions, observational_samples, coverage_total, graph, 
					num_additional_observations, costs, full_observational_samples, task = 'min', max_N = 200, 
					initial_num_obs_samples =100, num_interventions=10, Causal_prior=False, interventional_data=None):

        target_function_list = [None] * len(exploration_set)
        space_list = [None] * len(exploration_set)
        for s in range(len(exploration_set)):
            target_function_list[s], space_list[s] = Intervention_function(get_interventional_dict(exploration_set[s]),
                model = graph.define_SEM(), target_variable = 'Y',
                min_intervention = list_interventional_ranges(graph.get_interventional_ranges(), exploration_set[s])[0],
                max_intervention = list_interventional_ranges(graph.get_interventional_ranges(), exploration_set[s])[1]) 
        
        #? multiprocessing，packle，
        #? ，__main__
        # for s in range(len(exploration_set)):
        #     ES_i = interventional_data[s]
        #     x = ES_i[-2]
        #     y_label = np.empty_like(ES_i[-1]) #[20/100/1000, 1]
        #     y_label[:] = 0.  # used for Debug recheck
        #     pool = multiprocessing.Pool()
        #     results = [pool.apply_async(func=target_function_list[s], args=(x_i[np.newaxis,:],)) for x_i in x]
        #     # fs = [pool.apply_async(func=f, args=(i,)) for i in range(1, repeat)]
        #     pool.close()
        #     pool.join()
        #     for i, r in enumerate(results):
        #         y_label[i] = r.get()[0]
        #     interventional_data[s][-1] = y_label

        # # pathos.multiprocessing
        # for s in range(len(exploration_set)):
        #     ES_i = interventional_data[s]
        #     x = ES_i[-2]
        #     y_label = np.empty_like(ES_i[-1]) #[20/100/1000, 1]
        #     y_label[:] = 0.  # used for Debug recheck
        #     pool = Pool(20)
        #     # results = pool.map_async(target_function_list[s], [x_i[np.newaxis,:] for x_i in x])
        #     results = [pool.apply_async(func=target_function_list[s], args=(x_i[np.newaxis,:],)) for x_i in x]
        #     # fs = [pool.apply_async(func=f, args=(i,)) for i in range(1, repeat)]
        #     pool.close()
        #     pool.join()
        #     for i, r in enumerate(results):
        #         y_label[i] = r.get()[0]
        #     interventional_data[s][-1] = y_label

        # 
        for s in range(len(exploration_set)):
            ES_i = interventional_data[s]
            x = ES_i[-2]
            y_label = np.array([target_function_list[s](x_i[np.newaxis,:],)[0] for x_i in x])
            interventional_data[s][-1] = y_label

        tool.algorithm.dumpVariPickle(interventional_data, name='Data/CoralGraph/intervention_data_latest.pkl')

def cal_observation(num_trials, exploration_set, manipulative_variables, data_x_list, data_y_list,  best_intervention_value, opt_y, 
					best_variable, dict_ranges, functions, observational_samples, coverage_total, graph, 
					num_additional_observations, costs, full_observational_samples, task = 'min', max_N = 200, 
					initial_num_obs_samples =100, num_interventions=10, Causal_prior=False):
        target_function_list = [None]
        space_list = [None]
        target_function_list[0], space_list[0] = Intervention_function(get_interventional_dict(manipulative_variables),
                model = graph.define_SEM(), target_variable = 'Y',
                min_intervention = list_interventional_ranges(graph.get_interventional_ranges(), manipulative_variables)[0],
                max_intervention = list_interventional_ranges(graph.get_interventional_ranges(), manipulative_variables)[1]) 
        data = full_observational_samples._values
        y_label = np.empty(shape=[data.shape[0],1])
        for i in tqdm(range(data.shape[0])):
            y_label[i] = target_function_list[0](data[i,:][np.newaxis,:-1],)[0]
        tool.algorithm.dumpVariPickle(vari=y_label, name='Data/observation_latest.pkl')
        # full_observational_samples[:, -1] = y_label[:, -1]

        # tool.algorithm.dumpVariPickle(vari=full_observational_samples, name='Data/observation_latest.pkl')

def intervention_building(exploration_set:Tuple[List[str], List[List[str]]], graph, num:int=100, fname='Data/CoralGraph/intervention_CausalDimensionalSet'):
    '''CausalDimensionalSet(a list of startegy s),,BO
    '''
    intervention = []
    if not isinstance(exploration_set[0], List):
        exploration_set = [exploration_set]
    target_function_list = [None] * len(exploration_set)
    space_list = [None] * len(exploration_set)
    for i, s in enumerate(exploration_set):
        # X()
        max_intervention = np.array(list_interventional_ranges(graph.get_interventional_ranges(), s)[1])
        min_intervention = np.array(list_interventional_ranges(graph.get_interventional_ranges(), s)[0])
        
        # X
        intervention_X = np.random.rand((num), len(s))
        intervention_X = intervention_X * (max_intervention - min_intervention) + min_intervention
        
        # Evaluation
        target_function_list[i], space_list[i] = Intervention_function(get_interventional_dict(s),
                model = graph.define_SEM(), target_variable = 'Y',
                min_intervention = list_interventional_ranges(graph.get_interventional_ranges(), s)[0],
                max_intervention = list_interventional_ranges(graph.get_interventional_ranges(), s)[1])    
        intervention_Y = np.array([target_function_list[i](x_i[np.newaxis,:],)[0] for x_i in intervention_X])
        intervention_i = [len(s)]
        for si in s:
            intervention_i.append(si)
        intervention_i.append(intervention_X)
        intervention_i.append(intervention_Y)
        intervention.append(intervention_i)
    tool.algorithm.dumpVariPickle(vari=intervention, name=fname)
    # return intervention



def cross_val_score(num_trials, exploration_set, manipulative_variables, data_x_list, data_y_list,  best_intervention_value, opt_y, 
					best_variable, dict_ranges, functions, observational_samples, coverage_total, graph, 
					num_additional_observations, costs, full_observational_samples, task = 'min', max_N = 200, 
					initial_num_obs_samples =100, num_interventions=10, Causal_prior=False, 
                    interventional_data=None, index=-1, n_splits=10):

        # creat oracle
        # target_function_list = [None] * len(exploration_set)
        # space_list = [None] * len(exploration_set)
        model_list = [None] * n_splits
        # for s in range(len(exploration_set)):
        #     target_function_list[s], space_list[s] = Intervention_function(get_interventional_dict(exploration_set[s]),
        #         model = graph.define_SEM(), target_variable = 'Y',
        #         min_intervention = list_interventional_ranges(graph.get_interventional_ranges(), exploration_set[s])[0],
        #         max_intervention = list_interventional_ranges(graph.get_interventional_ranges(), exploration_set[s])[1]) 
                
        # calculate causal prior
        functions = graph.refit_models(full_observational_samples)
        current_best_x, current_best_y, x_dict_mean, x_dict_var, dict_interventions = initialise_dicts(exploration_set, task)
        mean_functions_list, var_functions_list = update_all_do_functions(graph, exploration_set, functions, dict_interventions, observational_samples, x_dict_mean, x_dict_var)
        
        # K
        X = interventional_data[index][-2]
        y = np.squeeze(interventional_data[index][-1])
        kfold = KFold(n_splits=n_splits, random_state=1, shuffle=True).split(X, y)
        metrics = np.empty(shape=(n_splits, 5))
        for i, (train_index, test_index) in enumerate(kfold):
            X_train, X_test = X[train_index], X[test_index] # len(train):len(test)  = 900:100
            y_train, y_test = y[train_index][:,np.newaxis], y[test_index][:,np.newaxis]
            model_list[i] = update_BO_models(mean_functions_list[index], var_functions_list[index], X_train, y_train, Causal_prior)
            y_mean, _ = model_list[i].predict(X_test)
            metrics[i] = metrics_GP(y=y_mean, y_label=y_test)
        print('means {}'.format(np.mean(metrics, axis=0)))

def plot_1D_mean_std():
    pass

def cross_var_score_RF(observational_samples, full_observational_samples, 
                    Causal_prior=False, #  
                    interventional_data=None, index=-1, n_splits=10):

    # K
    model_list = [None] * n_splits
    X = interventional_data[index][-2]  # 3
    y = np.squeeze(interventional_data[index][-1])
    kfold = KFold(n_splits=n_splits, random_state=1, shuffle=True).split(X, y)
    metrics = np.empty(shape=(n_splits, 5))
    for i, (train_index, test_index) in enumerate(kfold):
        X_train, X_test = X[train_index], X[test_index] # len(train):len(test)  = 900:100
        y_train, y_test = y[train_index][:,np.newaxis], y[test_index][:,np.newaxis]
        model_list[i] = RandomForestRegressor(oob_score=True, random_state=10)
        model_list[i].fit(X_train, y_train.ravel())
        y_mean = model_list[i].predict(X_test)
        metrics[i] = metrics_GP(y=y_mean, y_label=y_test)
    print('means {}'.format(np.mean(metrics, axis=0)))

def plot_y_1DX(interventional_data, index):
    X = interventional_data[index][-2]  # 1
    y = np.squeeze(interventional_data[index][-1])
    plt.plot(X, y, c='red')
    fname = os.path.join('figture_CoralGraph_CausalFalse_demo', 'plot_y_X', 'index'+str(index))
    plt.savefig(fname=fname)
    plt.cla()
    plt.close
    
def getXYZ(xstart=0, xend=1, ystart=0, yend=0, interval=10, bits=1, resource_ratio=0.5):
    problem2 = MOSG(player_num=2, target_num=2, resource_ratio=resource_ratio)
    x1 = np.linspace(xstart, xend, interval)
    x2 = np.linspace(ystart, yend, interval)
    x1_str = [str(round(i,bits)) for i in x1]
    x2_str = [str(round(i,bits)) for i in x2]
    # ，1x1,x22
    X1, X2 = np.meshgrid(x1, x2)
    payoff_attacker = np.empty_like(X1)
    payoff_defender = np.empty_like(X1)
    for i in range(len(x1)):
        for j in range(len(X2)):
            if x1[i]+x2[j]>problem2.resource:
                payoff_attacker[i, j] = 10
                payoff_defender[i, j] = 10
                continue

            problem2.set_ct(np.array([x1[i], x2[j]]))
            problem2.cal_payoff()
            problem2.cal_payoff_attacker()
            problem2.cal_payoff_defender()
            payoff_attacker[i, j] = problem2.get_payoff_attacker()
            payoff_defender[i, j] = problem2.get_payoff_defender()
    payoff_attacker[payoff_attacker==10] = math.floor(np.min(payoff_attacker))
    payoff_defender[payoff_defender==10] = math.floor(np.min(payoff_defender))
    return x1_str, x2_str, x1, x2, payoff_attacker, payoff_defender

''''''
def heatmap(data, row_labels=None, col_labels=None, ax=None,
            cbar_kw={}, cbarlabel="", 
            label_fontsize=None,
            **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    # plt.rc('font',  family='Times New Roman')
    plt.rc('font', family='serif') # NOTE serif
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    if label_fontsize is not None:
        ax.set_xticks(np.arange(data.shape[1]), labelsize=label_fontsize)
        ax.set_yticks(np.arange(data.shape[0]), labelsize=label_fontsize)
    else:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if col_labels is not None:
        ax.set_xticklabels(col_labels)
    if row_labels is not None:
        ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

''''''
def heatmapSG(data, row_labels=None, col_labels=None, ax=None,
            cbar_kw={}, cbarlabel="", 
            label_fontsize = 12, legend_fontsize=12,
            **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom") # colorbar，

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1])) # NOTE settick，tick（param）
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if col_labels is not None:
        ax.set_xticklabels(col_labels, fontsize=legend_fontsize) # set
    if row_labels is not None:
        ax.set_yticklabels(row_labels, fontsize=legend_fontsize)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False, 
                   labelsize=label_fontsize)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False,
                labelsize=label_fontsize)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, 
                     table_fontsize = None,
                     **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            if table_fontsize is not None:
                text = im.axes.text(j, i, valfmt(data[i, j], None), fontsize=table_fontsize, **kw)
            else:
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

'''T=2 N=2payoff of attacker and defender'''
def plot_y_2DX(interventional_data, index, pdf=False, interval=10, ):
    # 
    X = interventional_data[index][-2]  # 2
    y:np.ndarray = interventional_data[index][-1]
    y_idx = sorted(enumerate(X), key=lambda X:(X[0], X[1]))
    y_idx = [item[0] for item in y_idx]
    y_metrix = np.empty(shape=[interval, interval])
    y_metrix[:] = y[y_idx].reshape(y_metrix.shape)
    # x1_str, x2_str, x1, x2, payoff_attacker, payoff_defender = getXYZ(xend=1, yend=1, interval=interval+1, bits=2, resource_ratio=0.5) # 


    # fig, ax = plt.subplots(figsize=(7,8))
    fig, ax = plt.subplots()
    im, cbar = heatmap(y_metrix, ax=ax,)
    # texts = annotate_heatmap(im, valfmt="{x:.1f}", table_fontsize=table_fontsize)  # NOTE heatmap
    texts = annotate_heatmap(im, valfmt="{x:.1f}")
    # texts = annotate_heatmap(im, valfmt="{x:.1f} t")  # t
    fig.tight_layout()
    fname = os.path.join('figture_CoralGraph_CausalFalse_demo', 'plot_y_X', 'index'+str(index))
    plt.savefig(fname=fname)
    plt.cla()
    plt.close
    # if pdf:
    #     plt.savefig(fname=fname_pdf)

''''''
def f(x):
    # time.sleep(2)
    return ((((x**0.5+1)**2-1)**0.5+1)**2-1)**0.5 + 1
def test():
        time1 = time.time()
        repeat = 30
        result1 = []
        for i in range(1, repeat):
            result1.append(f(i))
        time2 = time.time()
        result1 = np.array(result1)
        print('',time2-time1)
        time1 = time.time()
        pool = multiprocessing.Pool()
        result2 = np.empty_like(result1)
        fs = [pool.apply_async(func=f, args=(i,)) for i in range(1, repeat)]
        pool.close()
        pool.join()
        for i, f_i in enumerate(fs):
            result2[i] = f_i.get()
        time2 = time.time()
        print('1',time2-time1)

        if np.array_equal(result1, result2):
            print('True')
        else:
            print(result1)
            print(result2)
# if __name__ == "__main__":
#         test()
'''Ktest'''
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([0, 0, 1, 1])
# skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1).split(X, y)
# #c= skf.get_n_splits(X, y)

# for train_index, test_index in skf:
#      print("TRAIN:", train_index, "TEST:", test_index)
#      X_train, X_test = X[train_index], X[test_index]
#      y_train, y_test = y[train_index], y[test_index]


if __name__ == '__main__':
    kernel_list = GPy.kern.BasisFuncKernel
    a = 1