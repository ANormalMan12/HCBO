from .draw_utils import *
def draw_graph(
        adjacency_matrix,
        title:str,
        manu_var_num=None,
        num_name:list=None):
    fig,ax=plt.subplots()
    G = nx.DiGraph()
    G.add_nodes_from(range(len(adjacency_matrix)))
    color_map=[]
    for i in range(len(adjacency_matrix)):
        G.nodes[i]['label'] =num_name[i] if num_name is not None else str(i)
        color_map.append('green')
        if(manu_var_num is not None and i in manu_var_num):
            color_map[i]='red'

    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G.add_edges_from(edges)
    topological_order = list(nx.topological_sort(G))
    layer_mapping = {node: topological_order.index(node) for node in topological_order}
    pos = nx.shell_layout(G, nlist=[list(filter(lambda x: layer_mapping[x] == i, G.nodes())) for i in range(len(set(layer_mapping.values())))])
    nx.draw(G, pos, node_size=200,node_color=color_map,arrows=True,ax=ax)
    labels=nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="black",ax=ax)
    ax.set_title(title)
    return fig