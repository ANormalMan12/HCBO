import os
import pickle
import json
import datetime
import functools
import numpy as np
from functools import reduce
from typing import *
import pandas as pd
def get_coral_graph_data():
    return {
        "observation":pd.read_csv('data/real_data/CoralGraph/observations.csv'),
        "true_observation":pd.read_csv('data/real_data/CoralGraph/true_observations.csv'),
    }
def get_protein_graph_data():
    return {
        "true_observation":pd.read_pickle('data/real_data/ProteinGraph/true_observations.pkl'),
    }
def dumpVariPickle(vari: Any, path:str=None, name:str=None):
    # kwargs.key , kwargs.value 
    # path.key , path.value 
    if path is None:
        path = os.getcwd()
        path = os.path.join(path, name)
    elif name is not None:
        path = os.path.join(path, name)
    with open(path, 'wb') as f:
        pickle.dump(vari, f)
        f.close()
    print('：', path)
def loadVariPickle(path:str) ->Any:
    # path.key = vari_name, path.value = vari
    print('：', path)
    with open(path, 'rb') as f:
        para = pickle.load(f)
        f.close()
        return para
# NOTE Json

def dumpVariJson(vari: Any, path:str=None, name:str=None, indent=4):
    # kwargs.key , kwargs.value 
    # path.key , path.value 
    if path is None:
        if not os.path.isabs(name):
            path = os.getcwd()
            path = os.path.join(path, name)
        else:
            path = name
    elif name is not None:
        if isinstance(path, list):
            path_i = os.getcwd()
            for dir in path:
                path_i = os.path.join(path_i, dir)
            path = path_i
        path = os.path.join(path, name)
    with open(path, 'w') as f:
        json.dump(vari, f, indent=indent)
        f.close()
    print('：', path)
def loadVariJson(path:str=None, name:str=None) ->Any:
    # path.key = vari_name, path.value = vari
    if path is None:
        path = name
    if isinstance(path, list):
        path_i = os.getcwd()
        for dir in path:
            path_i = os.path.join(path_i, dir)
        path = path_i
    print('：', path)
    with open(path, 'r') as f:
        data = f.read()
        para = json.loads(data)
        f.close()
        return para