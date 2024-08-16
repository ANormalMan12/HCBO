# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import numpy as np
# import gym
import json
import os
import random
from core import ObjFunc
#import imageio
import torch

class Levy:
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[0]) ** 2 + \
            np.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dim - 1] + 1) ** 2)) + \
            (w[self.dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dim - 1])**2)
        return val



class Ackley:
    def __init__(self, random_list, dims=10):
        self.dims      = dims
        self.lb        = -32.768 * np.ones(dims)
        self.ub        =  32.768 * np.ones(dims)
        self.random_list = random_list
        
        
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        #result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
        ObjFunc_x = torch.from_numpy(x)    #sampletensor
        ObjFunc_x = ObjFunc_x.reshape(1, self.dims)
        result = ObjFunc.ackley_rand(ObjFunc_x, self.random_list)
        result = np.float64(result)
           
        return -result   #


class Griewank:
    def __init__(self, random_list, dims=10):
        self.dims      = dims
        self.lb        = -50 * np.ones(dims)
        self.ub        =  50 * np.ones(dims)
        self.random_list = random_list
        
        
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        ObjFunc_x = torch.from_numpy(x)    #sampletensor
        ObjFunc_x = ObjFunc_x.reshape(1, self.dims)
        result = ObjFunc.griewank_rand(ObjFunc_x, self.random_list)
        result = np.float64(result)
                
        return -result




class Sphere:
    def __init__(self, random_list, dims=10):
        self.dims      = dims
        self.lb        = -5.12 * np.ones(dims)
        self.ub        =  5.12 * np.ones(dims)
        self.random_list = random_list
        
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        ObjFunc_x = torch.from_numpy(x)    #sampletensor
        ObjFunc_x = ObjFunc_x.reshape(1, self.dims)
        result = ObjFunc.sphere_rand(ObjFunc_x, self.random_list)
        result = np.float64(result)
                
        return -result
    

class Zakharov:
    def __init__(self, random_list, dims=10):
        self.dims      = dims
        self.lb        = -5. * np.ones(dims)
        self.ub        =  10. * np.ones(dims)
        self.random_list = random_list
        
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        ObjFunc_x = torch.from_numpy(x)    #sampletensor
        ObjFunc_x = ObjFunc_x.reshape(1, self.dims)
        result = ObjFunc.zakharov_rand(ObjFunc_x, self.random_list)
        result = np.float64(result)
                
        return -result


class Rosenbrock:
    def __init__(self, random_list, dims=10):
        self.dims = dims
        self.lb = -5. * np.ones(dims)
        self.ub = 10. * np.ones(dims)
        self.random_list = random_list

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        ObjFunc_x = torch.from_numpy(x)  # sampletensor
        ObjFunc_x = ObjFunc_x.reshape(1, self.dims)
        result = ObjFunc.rosenbrock_rand(ObjFunc_x, self.random_list)
        result = np.float64(result)

        return -result
    

class Cassini2Gtopx:
    def __init__(self, random_list, dims=22):
        self.dims = dims
        self.lb = np.array([-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01,
                            1.05, 1.05, 1.15, 1.7, -np.pi, -np.pi, -np.pi, -np.pi])
        self.ub = np.array([0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0,
                            6.5, 291.0, np.pi, np.pi, np.pi, np.pi])
        self.random_list = random_list

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        ObjFunc_x = torch.from_numpy(x)  # sampletensor
        ObjFunc_x = ObjFunc_x.reshape(1, self.dims)
        result = ObjFunc.cassini2_gtopx(ObjFunc_x, self.random_list)
        result = np.float64(result)

        return -result
    
    
