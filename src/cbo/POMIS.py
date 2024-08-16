from .npsem.model import *
from .npsem.scm_bandits import *
from .graph.CoralGraph.CoralGraph import CoralGraph
from .graph.ProteinGraph.ProteinGraph import ProteinGraph
from .utils import *
def getPOMIS_CoralGraph():
    data_dict=get_coral_graph_data()
    graph=CoralGraph(
        data_dict["observation"],
        data_dict["true_observation"]
    )
    edge_list=[]
    for i in range(graph.connection.shape[0]):
        for j in range(graph.connection.shape[1]):
            if(graph.connection[i][j]==1):
                edge_list.append((graph.num_name[i],graph.num_name[j]))
    return CausalDiagram(graph.num_name, edge_list,with_do=graph.get_sets()['manu_var'])


def getPOMIS_ProteinGraph():
    data_dict=get_protein_graph_data()
    graph=ProteinGraph(
        data_dict["true_observation"]
    )
    edge_list=[]
    for i in range(graph.connection.shape[0]):
        for j in range(graph.connection.shape[1]):
            if(graph.connection[i][j]==1):
                edge_list.append((graph.num_name[i],graph.num_name[j]))
    return CausalDiagram(graph.num_name, edge_list,with_do=set(graph.get_sets()['manu_var']))

def test_generate_POMIS():
    #G= getPOMIS_ProteinGraph()
    G=getPOMIS_CoralGraph()
    #Y: str="Y"
    pomiss = POMISs(G,'Y')
    miss = MISs(G, 'Y')
    all_ISs = {frozenset(xx) for xx in combinations(G.V - {'Y'})}
    print(f'{len(all_ISs)} ISs')
    print(f'{len(miss)} MISs')
    print(f'{len(pomiss)} POMISs')
    print(f'Brute-force arms: {sum([2**(len(iset)) for iset in all_ISs])}')
    print(f'        MIS arms: {sum([2**(len(mis)) for mis in miss])}')
    print(f'      POMIS arms: {sum([2**(len(pomis)) for pomis in pomiss])}')

    print('POMISs')
    for _, pomis in sorted([(len(pomis), tuple(sorted(pomis))) for pomis in pomiss]):
        print('  {', end='')
        print(*list(pomis), sep=', ', end='')
        print('}')

    print('MISs (but not POMISs)')
    for _, mis in sorted([(len(mis), tuple(sorted(mis))) for mis in miss - pomiss]):
        print('  {', end='')
        print(*list(mis), sep=', ', end='')
        print('}')

    #print('ISs (but not MISs)')
    #for _, iset in sorted([(len(iset), tuple(sorted(iset))) for iset in all_ISs - miss]):
    #    print('  {', end='')
    #    print(*list(iset), sep=', ', end='')
    #    print('}')
    #print(pomiss)