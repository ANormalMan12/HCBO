from .function import *
class NodeFunctionLayerConstructor():
    def __init__(self,layer:NodeFunctionLayer):
        self.layer=layer
    def add_multiply_operation(self,local_index_list:List[int]):
        self.layer.add_new_feature_action(MultiplyFunction(local_index_list))
    def add_single_operation(self,type:str,local_index):
        if(type=="sin"):
            self.layer.add_new_feature_action(SinFunction(local_index))
        elif(type=="cos"):
            self.layer.add_new_feature_action(CosFunction(local_index))
        elif(type=="ln"):
            self.layer.add_new_feature_action(LnFunction(local_index))
        elif(type=="same"):
            self.layer.add_new_feature_action(SameFunction(local_index))
        else:
            raise ValueError("The type of operation is not supported")
    def add_linear(self,weight_array:np.array,bias:np.array):
        self.layer.add_new_feature_action(LinearFunction(weight_array,bias))
    def add_linear_regression(self,input_obs,now_node_obs):
        action=LinearRegressionFunction(input_obs,now_node_obs)
        self.layer.add_new_feature_action(action)
    def get_layer(self):
        return self.layer


