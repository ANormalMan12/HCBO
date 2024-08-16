from typing import Any
from .constructor import *
#from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
class NodeFunction():
    def __init__(
            self,
            node_function_layer_list:List[NodeFunctionLayer],
            #exogenous_variable_distribution:Distribution
        ):
        #self.exogenous_variable_distribution=exogenous_variable_distribution
        assert(node_function_layer_list[-1].get_output_feature_size()==1)
        self.input_size=node_function_layer_list[0].input_size
        for i in range(len(node_function_layer_list)-1):
            assert(node_function_layer_list[i].get_output_feature_size()==node_function_layer_list[i+1].input_size)
        self._node_function_layer_list=node_function_layer_list
    def __call__(self,X):
        assert(self.input_size==X.shape[1])
        for node_function_layer in self._node_function_layer_list:
            X=node_function_layer(X)
        return X
class ZeroNodeFunction(NodeFunction):
    def __init__(self):
        pass
    def __call__(self,X):
        return np.zeros((X.shape[0],1))

from .synt_node import get_synt_non_additive_function_layer,get_synt_additive_function_layer

def __get_random_node_weight(input_size):
    ret_list=[]
    for i in range(input_size):
        if(np.random.uniform()<0.5):
            ret_list.append(np.random.uniform(0.5,1))
        else:
            ret_list.append(np.random.uniform(1,2))
    return np.array(ret_list)
import copy
def get_linear_layer(linear_input_size):
    linear_layer_worker = NodeFunctionLayerConstructor(
        layer=NodeFunctionLayer(
            input_size=linear_input_size
        )
    )
    linear_layer_worker.add_linear(
        weight_array=__get_random_node_weight(input_size=linear_input_size),
        bias=np.random.uniform(-1,1)
    )
    linear_layer=linear_layer_worker.get_layer()
    return linear_layer
class NodeFunctionBuilder():
    def __init__(self,input_size:int,node_type:str):
        self.input_size=input_size
        if(self.input_size==0):
            return
        self.node_type=node_type
        self.node_function_layer_list=[]
        if(node_type=="linear"):
            pass
        elif(node_type=="additive"):
            self.node_function_layer_list.append(
                get_synt_additive_function_layer(input_size)
            )
        elif(node_type=="non-additive"):
            self.node_function_layer_list.append(
                get_synt_non_additive_function_layer(input_size)
            )
        else:
            raise ValueError("node_type not supported")
        

    def get_oracle_node(self):
        """This function gives random initailized weights for node"""
        if(self.input_size==0):
            return ZeroNodeFunction()
        tmp_node_function_layer_list=copy.deepcopy(self.node_function_layer_list)
        if(len(self.node_function_layer_list)!=0):
            linear_input_size=self.node_function_layer_list[-1].get_output_feature_size()
        else:
            linear_input_size=self.input_size
        
        return NodeFunction(
            node_function_layer_list=tmp_node_function_layer_list+[get_linear_layer(linear_input_size)]
        )
    def get_fitted_node(self,input_observation,now_node_observation):
        if(self.input_size==0):
            return ZeroNodeFunction()
        if(len(self.node_function_layer_list)!=0):
            fitted_input_size=self.node_function_layer_list[-1].get_output_feature_size()
        else:
            fitted_input_size=self.input_size
        transform_input_observation=input_observation
        for node_function in self.node_function_layer_list:
            transform_input_observation=node_function(transform_input_observation)
        
        assert(fitted_input_size==transform_input_observation.shape[1])
        assert(transform_input_observation.shape[0]==now_node_observation.shape[0])
        #assert(now_node_observation.shape[1]==1)
        linear_reg_layer=NodeFunctionLayer(
            input_size=fitted_input_size
        )
        to_add_action=self.fit_method()(
            obs_X=transform_input_observation,
            obs_now=now_node_observation
        )
        linear_reg_layer.add_new_feature_action(
            to_add_action
        )
        return NodeFunction(
            node_function_layer_list=self.node_function_layer_list+[linear_reg_layer]
        )
    def fit_method(self):
        return LinearRegressionFunction
    
class NodeFunctionBuilderGP(NodeFunctionBuilder):
    def fit_method(self):
        return GaussianProcessRegressionFunction