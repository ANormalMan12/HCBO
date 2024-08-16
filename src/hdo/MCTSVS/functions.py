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
# import ObjFunc
#import imageio
import torch

class tracker:
    def __init__(self, foldername):
        self.counter   = 0
        self.results   = []
        self.curt_best = -float("inf")
        self.foldername = foldername
        print(foldername)
        try:
            os.mkdir(foldername)
        except OSError:
            print ("Creation of the directory %s failed" % foldername)
        else:
            print ("Successfully created the directory %s " % foldername)
        
    def dump_trace(self):
        trace_path = self.foldername + '/result' + str(len( self.results) )
        final_results_str = json.dumps(self.results)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')
            
    def track(self, result):
        if result > self.curt_best:
            self.curt_best = result
        self.results.append(self.curt_best)
        if len(self.results) % 100 == 0:
            self.dump_trace()


class Levy:
    def __init__(self, dims=10):
        self.dims        = dims
        self.lb          = -10 * np.ones(dims)
        self.ub          =  10 * np.ones(dims)
        self.tracker     = tracker('Levy'+str(dims))
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp          = 10
        self.leaf_size   = 8
        self.kernel_type = "poly"
        self.ninits      = 40
        self.gamma_type   = "auto"
        print("initialize levy at dims:", self.dims)
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        w = []
        for idx in range(0, len(x)):
            w.append( 1 + (x[idx] - 1) / 4 )
        w = np.array(w)
        
        term1 = ( np.sin( np.pi*w[0] ) )**2
        
        term3 = ( w[-1] - 1 )**2 * ( 1 + ( np.sin( 2 * np.pi * w[-1] ) )**2 )
        
        
        term2 = 0
        for idx in range(1, len(w) ):
            wi  = w[idx]
            new = (wi-1)**2 * ( 1 + 10 * ( np.sin( np.pi* wi + 1 ) )**2)
            term2 = term2 + new
        
        result = term1 + term2 + term3
        self.tracker.track( result )

        return result



#class Ackley:
    # def __init__(self, dims=10):
    #     self.dims      = dims
    #     self.lb        = -5 * np.ones(dims)
    #     self.ub        =  10 * np.ones(dims)
    #     self.counter   = 0
    #     self.tracker   = tracker('Ackley'+str(dims) )
    #     
    #     #tunable hyper-parameters in LA-MCTS
    #     self.Cp        = 1
    #     self.leaf_size = 10
    #     self.ninits    = 40
    #     self.kernel_type = "rbf"
    #     self.gamma_type  = "auto"
    #     
    #     
    # def __call__(self, x):
    #     self.counter += 1
    #     assert len(x) == self.dims
    #     assert x.ndim == 1
    #     assert np.all(x <= self.ub) and np.all(x >= self.lb)
    #     result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
    #     #print(type(result))
    #     self.tracker.track( result )
    #             
    #     return result
    # 


class new_obj:
    def __init__(self, random_list, fold_name, init, dims=10,objFunction=None):
        if(ObjFunc==None):raise(ValueError("WRONG"))
        self.dims      = dims
        self.lb        = 0.05 * np.ones(dims)
        self.ub        = 0.95 * np.ones(dims)
        self.counter   = 0
        self.tracker   = tracker(fold_name)
        self.random_list = random_list
        self.objFunction=objFunction
        #tunable hyper-parameters in LA-MCTS
        self.Cp        = 10
        self.leaf_size = 10
        self.ninits    = init
        self.kernel_type = "rbf"
        self.gamma_type  = "auto"
        
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        # result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
        ObjFunc_x = torch.from_numpy(x)    #sampletensor
        ObjFunc_x = ObjFunc_x.reshape(-1, self.dims)
        result = self.objFunction(ObjFunc_x, self.random_list)
        result = np.float64(result)
        self.tracker.track( result )                
        return result

class Ackley:
    def __init__(self, random_list, fold_name, init, dims=10):
        self.dims      = dims
        self.lb        = -32.768 * np.ones(dims)
        self.ub        =  32.768 * np.ones(dims)
        self.counter   = 0
        self.tracker   = tracker(fold_name)
        self.random_list = random_list
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp        = 10
        self.leaf_size = 10
        self.ninits    = init
        self.kernel_type = "rbf"
        self.gamma_type  = "auto"
        
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        # result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
        ObjFunc_x = torch.from_numpy(x)    #sampletensor
        ObjFunc_x = ObjFunc_x.reshape(-1, self.dims)
        result = ObjFunc.ackley_rand(ObjFunc_x, self.random_list)
        result = np.float64(result)
        self.tracker.track( result )
                
        return result


class Griewank:
    def __init__(self, random_list, fold_name, init, dims=10):
        self.dims      = dims
        self.lb        = -50 * np.ones(dims)
        self.ub        =  50 * np.ones(dims)
        self.counter   = 0
        self.tracker   = tracker(fold_name)
        self.random_list = random_list
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp        = 10
        self.leaf_size = 10
        self.ninits    = init
        self.kernel_type = "rbf"
        self.gamma_type  = "auto"
        
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        ObjFunc_x = torch.from_numpy(x)    #sampletensor
        ObjFunc_x = ObjFunc_x.reshape(1, self.dims)
        result = ObjFunc.griewank_rand(ObjFunc_x, self.random_list)
        result = np.float64(result)
        self.tracker.track( result )
                
        return result




class Sphere:
    def __init__(self, random_list, fold_name, init, dims=10):
        self.dims      = dims
        self.lb        = -5.12 * np.ones(dims)
        self.ub        =  5.12 * np.ones(dims)
        self.counter   = 0
        self.tracker   = tracker(fold_name)
        self.random_list = random_list
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp        = 10
        self.leaf_size = 10
        self.ninits    = init
        self.kernel_type = "rbf"
        self.gamma_type  = "auto"
        
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        ObjFunc_x = torch.from_numpy(x)    #sampletensor
        ObjFunc_x = ObjFunc_x.reshape(1, self.dims)
        result = ObjFunc.sphere_rand(ObjFunc_x, self.random_list)
        result = np.float64(result)
        self.tracker.track( result )
                
        return result
    

class Zakharov:
    def __init__(self, random_list, fold_name, init, dims=10):
        self.dims      = dims
        self.lb        = -5.12 * np.ones(dims)
        self.ub        =  5.12 * np.ones(dims)
        self.counter   = 0
        self.tracker   = tracker(fold_name)
        self.random_list = random_list
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp        = 10
        self.leaf_size = 10
        self.ninits    = init
        self.kernel_type = "rbf"
        self.gamma_type  = "auto"
        
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        ObjFunc_x = torch.from_numpy(x)    #sampletensor
        ObjFunc_x = ObjFunc_x.reshape(1, self.dims)
        result = ObjFunc.zakharov_rand(ObjFunc_x, self.random_list)
        result = np.float64(result)
        self.tracker.track( result )
                
        return result


class Rosenbrock:
    def __init__(self, random_list, fold_name, init, dims=10):
        self.dims = dims
        self.lb = -5. * np.ones(dims)
        self.ub = 10. * np.ones(dims)
        self.counter = 0
        self.tracker = tracker(fold_name)
        self.random_list = random_list

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 10
        self.leaf_size = 10
        self.ninits = init
        self.kernel_type = "rbf"
        self.gamma_type = "auto"

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        ObjFunc_x = torch.from_numpy(x)  # sampletensor
        ObjFunc_x = ObjFunc_x.reshape(1, self.dims)
        result = ObjFunc.rosenbrock_rand(ObjFunc_x, self.random_list)
        result = np.float64(result)
        self.tracker.track(result)

        return result


class Cassini2Gtopx:
    def __init__(self, random_list, fold_name, init, dims=22):
        self.dims = dims
        self.lb = np.array([-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01,
                            1.05, 1.05, 1.15, 1.7, -np.pi, -np.pi, -np.pi, -np.pi])
        self.ub = np.array([0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0,
                            6.5, 291.0, np.pi, np.pi, np.pi, np.pi])
        self.counter = 0
        self.tracker = tracker(fold_name)
        self.random_list = random_list

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 10
        self.leaf_size = 10
        self.ninits = init
        self.kernel_type = "rbf"
        self.gamma_type = "auto"

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        ObjFunc_x = torch.from_numpy(x)  # sampletensor
        ObjFunc_x = ObjFunc_x.reshape(1, self.dims)
        result = ObjFunc.cassini2_gtopx(ObjFunc_x, self.random_list)
        result = np.float64(result)
        self.tracker.track(result)

        return result


# class Lunarlanding:
#     def __init__(self):
#         self.dims = 12
#         self.lb   = np.zeros(12)
#         self.ub   = 2 * np.ones(12)
#         self.counter = 0
#         self.env = gym.make('LunarLander-v2')
#
#         #tunable hyper-parameters in LA-MCTS
#         self.Cp          = 50
#         self.leaf_size   = 10
#         self.kernel_type = "poly"
#         self.ninits      = 40
#         self.gamma_type  = "scale"
#
#         self.render      = False
#
#
#     def heuristic_Controller(self, s, w):
#         angle_targ = s[0] * w[0] + s[2] * w[1]
#         if angle_targ > w[2]:
#             angle_targ = w[2]
#         if angle_targ < -w[2]:
#             angle_targ = -w[2]
#         hover_targ = w[3] * np.abs(s[0])
#
#         angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
#         hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]
#
#         if s[6] or s[7]:
#             angle_todo = w[8]
#             hover_todo = -(s[3]) * w[9]
#
#         a = 0
#         if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
#             a = 2
#         elif angle_todo < -w[11]:
#             a = 3
#         elif angle_todo > +w[11]:
#             a = 1
#         return a
#
#     def __call__(self, x):
#         self.counter += 1
#         assert len(x) == self.dims
#         assert x.ndim == 1
#         assert np.all(x <= self.ub) and np.all(x >= self.lb)
#
#         total_rewards = []
#         for i in range(0, 3): # controls the number of episode/plays per trial
#             state = self.env.reset()
#             rewards_for_episode = []
#             num_steps = 2000
#
#             for step in range(num_steps):
#                 if self.render:
#                     self.env.render()
#                 received_action = self.heuristic_Controller(state, x)
#                 next_state, reward, done, info = self.env.step(received_action)
#                 rewards_for_episode.append( reward )
#                 state = next_state
#                 if done:
#                      break
#
#             rewards_for_episode = np.array(rewards_for_episode)
#             total_rewards.append( np.sum(rewards_for_episode) )
#         total_rewards = np.array(total_rewards)
#         mean_rewards = np.mean( total_rewards )
#
#         return mean_rewards*-1
#

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
