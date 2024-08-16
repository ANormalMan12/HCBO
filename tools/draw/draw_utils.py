from matplotlib.axes import Axes
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']

def strize_exp_identifier(exp_identifier):
    return f"{exp_identifier[0]}-{exp_identifier[1]}-{exp_identifier[2]}"
MARKER_LIST = ['o', 's', 'D', 'v', '^', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', ',', '.', '1', '2', '3', '4']

RGB_LIST =[
    (68,4,90),
    (38,70,83),
    (40,114,113),
    (42,157,140),
    (138,176,125),
    (145,214,66),
    (233,196,107),
    (243,162,97),
    (230,111,81),
    (248,230,32)
]
COLOR_LIST=[
    (R/255.0,G/255.0,B/255.0)for (R,G,B)in RGB_LIST
]

def calculate_stats(column):
    min_value = column.min()
    max_value = column.max()
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    mean_value = column.mean()
    return pd.Series({'Min': min_value, 'Q1': q1, 'Mean': mean_value,'Q3': q3, 'Max': max_value})
def draw_data(data,ax:plt.Axes,label):
    means = np.mean(data, axis=0)
    std_devs = np.std(data, axis=0)
    confidence_intervals = 1.96 * (std_devs / np.sqrt(data.shape[0]))  # 95% 
    
    steps = np.arange(0,means.shape[0])

    ax.fill_between(steps, means+confidence_intervals, means-confidence_intervals, alpha=0.2)
    ax.plot(steps, means, label=label)
    #plt.errorbar(x=steps,y= means, yerr=confidence_intervals, label='Line '+str(key), fmt='-o', capsize=5)
