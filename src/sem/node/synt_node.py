from .constructor import *
import random
function_type_probabiity_list=[
    ["sin","cos","ln","same"],
    [0.05,0.05,0.1,0.8]# must sum up to 1
]

def get_synt_additive_function_layer(input_size):
    worker=NodeFunctionLayerConstructor(
        NodeFunctionLayer(input_size=input_size)
    )
    for i in range(input_size):
        eps=np.random.uniform()# between 0 and 1
        for j,val in enumerate(function_type_probabiity_list[1]):
            if(eps<val):
                now_type=function_type_probabiity_list[0][j]
            else:
                eps=eps-val
        if(now_type is None):now_type="same"
        worker.add_single_operation(type=now_type,local_index=i)#i is a local index
    return worker.get_layer()


def get_synt_non_additive_function_layer(input_size):
    """The function not only introduces non_additive function, but also uses
    get_synt_additive_function_layer to include additive function
    """
    worker=NodeFunctionLayerConstructor(
        get_synt_additive_function_layer(input_size)
    )
    if(input_size>=2):
        for k in range(input_size//6):
            local_index_list=random.sample(range(input_size),2)
            worker.add_multiply_operation(local_index_list)
    return worker.get_layer()

