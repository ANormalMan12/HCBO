# This file implements basic function that can be used in SCM


from typing import List
import numpy as np
from abc import ABC,abstractmethod
    
class AbstractSCMFunction(ABC):
    def __init__():
        pass
    @abstractmethod
    def __call__(self,X):
        raise NotImplementedError("This is an abstract method")

class MultiplyFunction(AbstractSCMFunction):
    def __init__(self,multiply_variables:list):
        self.multiply_variables=multiply_variables
    def __call__(self,X):
        result=np.ones(X.shape[0])
        sum_array=np.zeros(X.shape[0])#!
        for variable in self.multiply_variables:
            result*=X[:,variable]
            sum_array+=X[:,variable]
        result=result/(np.absolute(sum_array)+1)
        return result.reshape(X.shape[0])

class SinFunction(AbstractSCMFunction):
    def __init__(self,variable):
        self.variable=variable
    def __call__(self,X):
        return np.sin(X[:,self.variable]).reshape(X.shape[0])

class CosFunction(AbstractSCMFunction):
    def __init__(self,variable):
        self.variable=variable
    def __call__(self,X):
        return np.cos(X[:,self.variable]).reshape(X.shape[0])

class LnFunction(AbstractSCMFunction):
    def __init__(self,variable):
        self.variable=variable
    def __call__(self,X):
        return np.log(X[:,self.variable]).reshape(X.shape[0])

class SameFunction(AbstractSCMFunction):
    def __init__(self,variable):
        self.variable=variable
    def __call__(self,X):
        return X[:,self.variable].reshape(X.shape[0])

class LinearFunction(AbstractSCMFunction):
    def __init__(self,weight_array,bias):
        self.weight_array=weight_array
        self.bias=bias
    def __call__(self,X):
        return X@self.weight_array+self.bias

from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
class LinearRegressionFunction(AbstractSCMFunction):
    def __init__(self,obs_X,obs_now):
        self.reg=LinearRegression()
        self.reg.fit(obs_X,obs_now)
        y_pred = self.reg.predict(obs_X)
        mse = mean_squared_error(obs_now, y_pred)
        print("MSE:",mse)
        #print("Coefs:",self.reg.coef_)
    def __call__(self,X):
        return self.reg.predict(X).flatten()
class GaussianProcessRegressionFunction():
    def __init__(self,obs_X,obs_now):
        self.reg=GaussianProcessRegressor()
        self.reg.fit(obs_X,obs_now)
        y_pred = self.reg.predict(obs_X)
        mse = mean_squared_error(obs_now, y_pred)
        print("MSE:",mse)
    def __call__(self,X):
        return self.reg.predict(X).flatten()
    
class NodeFunctionLayer():
    def __init__(self,input_size:int):
        self.input_size=input_size
        self.action_list=[]
    def add_new_feature_action(self,action:AbstractSCMFunction):
        self.action_list.append(action)
    def get_output_feature_size(self):
        return len(self.action_list)
    
    def __call__(self,X):
        assert(X.shape[1]==self.input_size)
        new_feature_X=np.zeros((X.shape[0],len(self.action_list)))
        for i,action in enumerate(self.action_list):
            new_feature_X[:,i]=action(X)
        return new_feature_X
