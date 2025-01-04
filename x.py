import rpy2.robjects as ro
from rpy2.robjects import numpy2ri

import string
import random
import copy

import dgp

ro.r['source']('./load_pags.r')
load_pags = ro.globalenv['load_pags']

truePAGs, subsetsList = load_pags()

subsetsList = [(sorted(tuple(x[0])), sorted(tuple(x[1]))) for x in subsetsList]

def floatmatrix_to_2dlist(r_floatmatrix):
    numpy_matrix = numpy2ri.rpy2py(r_floatmatrix)
    return numpy_matrix.astype(int).tolist()
truePAGs = [floatmatrix_to_2dlist(pag) for pag in truePAGs]

# Adjacency Matrix Arrowheads:
# 0: Missing Edge
# 1: Dot Head
# 2: Arrow Head
# 3: Tail
def pag_to_node_collection(pag):
    alphabet = string.ascii_uppercase

    def get_node_collection(pag):
        nodes = []
        for i in range(len(pag)):
            nodes.append(dgp.Node(name=alphabet[i]))

        for i in range(len(pag)):
            for j in range(i, len(pag)):
                # Arrowhead on Node i
                marker_1 = pag[i][j]
                # Arrowhead on Node j
                marker_2 = pag[j][i]

                assert (marker_1 != 0 and marker_2 != 0) or marker_1 == marker_2, 'If one is 0, the other needs to be as well'

                # no edge
                if marker_1 == 0 or marker_2 == 0:
                    continue

                # Turn odot ends into tails
                marker_1 = 3 if marker_1 == 1 else marker_1
                marker_2 = 3 if marker_2 == 1 else marker_2

                # edges must have at least one arrow
                assert marker_1 != 3 or marker_2 != 3, 'If one is tail, the other can not be'

                assert marker_1 in [2,3] and marker_2 in [2,3], 'Only tails and arrows allowed after this point'

                ## start adding parents
                if marker_1 == 2 and marker_2 == 2:
                    # add latent confounder
                    # TODO: Maybe make this only continuos values
                    confounder = dgp.Node(name=f'L_{alphabet[i]}{alphabet[j]}')
                    nodes.append(confounder)
                    nodes[i].parents.append(confounder)
                    nodes[j].parents.append(confounder)
                elif marker_1 == 3 and marker_2 == 2:
                    nodes[i].parents.append(nodes[j])
                elif marker_1 == 2 and marker_2 == 3:
                    nodes[j].parents.append(nodes[i])
                else:
                    raise Exception('Two tails on one edge are not allowed at this point')
        nc = dgp.NodeCollection(
            name='test',
            nodes=nodes,
            drop_vars=[n.name for n in nodes[len(pag):]] # drop all vars outside the adjacency matrix -> confounders
        )
        return nc

    # Fix odot to odot edges by trying both
    def get_options_for_odot_edges(pag):
        pags = []
        for i in range(len(pag)):
            for j in range(i, len(pag)):
                # Arrowhead on Node i
                marker_1 = pag[i][j]
                # Arrowhead on Node j
                marker_2 = pag[j][i]

                if marker_1 == 1 and marker_2 == 1:
                    _pag_1 = copy.deepcopy(pag)
                    _pag_2 = copy.deepcopy(pag)
                    _pag_1[i][j] = 2
                    _pag_2[j][i] = 2
                    pags.extend(get_options_for_odot_edges(_pag_1))
                    pags.extend(get_options_for_odot_edges(_pag_2))
                    return pags
        return [pag]

    pags = get_options_for_odot_edges(pag)
    ncs = []
    for pag in pags:
        try:
            nc = get_node_collection(pag)
            ncs.append(nc)
        except:
            continue
    assert len(ncs) > 0, 'At least one result is required'
    nc = random.choice(ncs)
    return nc.reset()

# TODO: create node collection from scratch every time to avoid always same structure for double odot edges
ncs = [pag_to_node_collection(pag) for pag in truePAGs]


test_setups = list(zip(truePAGs, subsetsList))

test_setup = test_setups[0]
nc = pag_to_node_collection(test_setup[0])
client_A_subset = test_setup[1][0]
client_B_subset = test_setup[1][1]

NUM_SAMPLES = 1000
CLIENT_A_DATA_FRACTION = 0.2

data = nc.get(NUM_SAMPLES)

split_point = int(CLIENT_A_DATA_FRACTION*NUM_SAMPLES)
client_A_data = data[:split_point].select(client_A_subset)
client_B_data = data[split_point:].select(client_B_subset)

print(client_A_data)
