import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def drawGraph(adjacency_matrix,num_name:list,note:str,manu_var=None,show_number:bool=False):
    plt.clf()
    G = nx.DiGraph()
    # 
    G.add_nodes_from(range(len(adjacency_matrix)))
    # 
    color_map=[]
    for i,name in enumerate(num_name):
        G.nodes[i]['label'] = name
        color_map.append('green')
        if(manu_var is not None and name in manu_var):
            color_map[i]='red'
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G.add_edges_from(edges)
    # 
    #------------------
    topological_order = list(nx.topological_sort(G))
    # 
    layer_mapping = {node: topological_order.index(node) for node in topological_order}
    #  shell_layout ，
    pos = nx.shell_layout(G, nlist=[list(filter(lambda x: layer_mapping[x] == i, G.nodes())) for i in range(len(set(layer_mapping.values())))])
    #-------------------
    nx.draw(G, pos, node_size=200,node_color=color_map,arrows=True)
    nx.draw_networkx_labels(G, pos, labels=None if show_number else nx.get_node_attributes(G, 'label'), font_size=10, font_color="black")

    plt.title(note)
    plt.show()