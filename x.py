import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri

import string
import random
import copy
import json
import itertools
from pathlib import Path
from collections import OrderedDict
import pandas as pd
import numpy as np
from tqdm import tqdm

import dgp
import fedci

# supress R log
import rpy2.rinterface_lib.callbacks as cb
cb.consolewrite_print = lambda x: None
cb.consolewrite_warnerror = lambda x: None

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


    # TODO: AVOID - NEW COLLIDERS    (done)
    #             - CYCLES           (done)
    #             - UNDIRECTED EDGES (done)

    # Fix odot to odot edges by trying both
    def get_options_for_odot_edges(true_pag, pag):
        pags = []
        for i in range(len(pag)):
            for j in range(i, len(pag)):
                # Arrowhead on Node i
                marker_1 = pag[i][j]
                # Arrowhead on Node j
                marker_2 = pag[j][i]

                if marker_1 == 1 and marker_2 == 1:
                    pag_array = np.array(pag)
                    _pag_1 = pag_array.copy()
                    if np.sum((_pag_1[:,j] == 2) * (true_pag[:,j] == 1)) == 0:
                        _pag_1[i,j] = 2
                        _pag_1[j,i] = 3
                        pags.extend(get_options_for_odot_edges(true_pag, _pag_1.tolist()))

                    _pag_2 = pag_array.copy()
                    if np.sum((_pag_2[:,i] == 2) * (true_pag[:,i] == 1)) == 0:
                        _pag_2[i,j] = 3
                        _pag_2[j,i] = 2
                        pags.extend(get_options_for_odot_edges(true_pag, _pag_2.tolist()))

                    _pag_3 = pag_array.copy()
                    if (np.sum((_pag_3[:,i] == 2) * (true_pag[:,i] == 1)) == 0) and \
                        (np.sum((_pag_3[:,j] == 2) * (true_pag[:,j] == 1)) == 0):
                        _pag_3[i,j] = 2
                        _pag_3[j,i] = 2
                        pags.extend(get_options_for_odot_edges(true_pag, _pag_3.tolist()))

                    return pags
        return [pag]

    pags = get_options_for_odot_edges(np.array(copy.deepcopy(pag)), copy.deepcopy(pag))
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

test_setups = list(zip(truePAGs, subsetsList))

NUM_SAMPLES = 1000
CLIENT_A_DATA_FRACTION = 0.2

def setup_servers(test_setup):
    pag = test_setup[0]
    nc = pag_to_node_collection(pag)
    client_A_subset = test_setup[1][0]
    client_B_subset = test_setup[1][1]

    data = nc.get(NUM_SAMPLES)

    split_point = int(CLIENT_A_DATA_FRACTION*NUM_SAMPLES)
    client_A_data = data[:split_point].select(client_A_subset)
    client_B_data = data[split_point:].select(client_B_subset)

    client_A = fedci.Client(client_A_data)
    client_B = fedci.Client(client_B_data)

    server_single =  fedci.Server(
        {
            "1": client_A
        }
    )

    server_coop = fedci.Server(
        {
            "1": client_A,
            "2": client_B
        }
    )

    return (pag, sorted(data.columns)), (server_single, server_coop)

def server_results_to_dataframe(labels, results):
    likelihood_ratio_tests = fedci.get_symmetric_likelihood_tests(results)

    columns = ('ord', 'X', 'Y', 'S', 'pvalue')
    rows = []

    lrt_ord_0 = [(lrt.v0, lrt.v1) for lrt in likelihood_ratio_tests if len(lrt.conditioning_set) == 0]
    label_combinations = itertools.combinations(labels, 2)
    missing_base_rows = []
    for label_combination in label_combinations:
        if label_combination in lrt_ord_0:
            continue
        #print('MISSING', label_combination)
        l0, l1 = label_combination
        missing_base_rows.append((0, labels.index(l0)+1, labels.index(l1)+1, "", 1))
    rows += missing_base_rows

    for test in likelihood_ratio_tests:
        s_labels_string = ','.join(sorted([str(labels.index(l)+1) for l in test.conditioning_set]))
        rows.append((len(test.conditioning_set), labels.index(test.v0)+1, labels.index(test.v1)+1, s_labels_string, test.p_val))

    df = pd.DataFrame(data=rows, columns=columns)
    return df

def run_riod(df, labels, client_labels, alpha):
    ro.r['source']('./aggregation.r')
    iod_on_ci_data_f = ro.globalenv['iod_on_ci_data']
    # Reading and processing data
    #df = pl.read_csv("./random-data-1.csv")

    # let index start with 1
    df.index += 1

    label_list = [ro.StrVector(v) for v in client_labels.values()]
    users = list(client_labels.keys())

    with (ro.default_converter + pandas2ri.converter).context():
        #converting it into r object for passing into r function
        suff_stat = [
            ('citestResults', ro.conversion.get_conversion().py2rpy(df)),
            ('all_labels', ro.StrVector(labels)),
        ]
        suff_stat = OrderedDict(suff_stat)
        suff_stat = ro.ListVector(suff_stat)

        result = iod_on_ci_data_f(label_list, suff_stat, alpha)

        g_pag_list = [x[1].tolist() for x in result['G_PAG_List'].items()]
        g_pag_labels = [list(x[1]) for x in result['G_PAG_Label_List'].items()]
        g_pag_list = [np.array(pag).astype(int).tolist() for pag in g_pag_list]
        gi_pag_list = [x[1].tolist() for x in result['Gi_PAG_list'].items()]
        gi_pag_labels = [list(x[1]) for x in result['Gi_PAG_Label_List'].items()]
        g_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]
        return g_pag_list, g_pag_labels, {u:r for u,r in zip(users, gi_pag_list)}, {u:l for u,l in zip(users, gi_pag_labels)}

def filter_adjacency_matrices(pag, pag_labels, filter_labels):
    # Convert to numpy arrays for easier manipulation
    pag = np.array(pag)

    # Find indices of pred_labels in true_labels to maintain the order of pred_labels
    indices = [pag_labels.index(label) for label in filter_labels if label in pag_labels]

    # Filter the rows and columns of true_pag to match the order of pred_labels
    filtered_pag = pag[np.ix_(indices, indices)]

    # Extract the corresponding labels
    filtered_true_labels = [pag_labels[i] for i in indices]

    return filtered_pag.tolist(), filtered_true_labels

def evaluate_prediction(true_pag, pred_pag, true_labels, pred_labels):
    shd = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    correct_edges = 0
    other = 0

    true_sub_pag, true_sub_labels = filter_adjacency_matrices(true_pag, true_labels, pred_labels)
    if len(pred_pag) > len(pred_labels):
        pred_pag, _ = filter_adjacency_matrices(pred_pag, true_labels, pred_labels)

    assert tuple(true_sub_labels) == tuple(pred_labels), 'When evaluating, subgraph of true PAG needs to match vertices of predicted PAG'

    for i in range(len(true_sub_pag)):
        for j in range(i, len(true_sub_pag)):
            true_edge_start = true_sub_pag[i][j]
            true_edge_end = true_sub_pag[j][i]

            assert (true_edge_start != 0 and true_edge_end != 0) or true_edge_start == true_edge_end, 'Missing edges need to be symmetric'

            pred_edge_start = pred_pag[i][j]
            pred_edge_end = pred_pag[j][i]

            assert (pred_edge_start != 0 and pred_edge_end != 0) or pred_edge_start == pred_edge_end, 'Missing edges need to be symmetric'

            # Missing edge in both
            if true_edge_start == 0 and pred_edge_start == 0:
                tn += 1
                continue

            # False Positive
            if true_edge_start == 0 and pred_edge_start != 0:
                fp += 1
                shd += 1
                continue

            # False Negative
            if true_edge_start != 0 and pred_edge_start == 0:
                fn += 1
                shd += 1
                continue
            # True Positive
            if true_edge_start != 0 and pred_edge_start != 0:
                tp += 1
                continue

            # Same edge in both
            if true_edge_start == pred_edge_start and true_edge_end == pred_edge_end:
                correct_edges += 1
                continue

            other += 1
            shd += 1

    return shd, tp, tn, fp, fn, other, correct_edges

def log_results(target_dir, target_file, results_s, results_c, results_s_local, results_c_local):
    if len(results_s) != 0 and len(results_c) != 0:
        shd_s, tp_s, tn_s, fp_s, fn_s, other_s, correct_edges_s = zip(*results_s)
        shd_c, tp_c, tn_c, fp_c, fn_c, other_c, correct_edges_c = zip(*results_c)

        false_discovery_rate_s = [_fp_s/(_fp_s+_tp_s) if _fp_s+_tp_s > 0 else 0 for _fp_s, _tp_s in zip(fp_s, tp_s)]
        false_omission_rate_s = [_fn_s/(_fn_s+_tn_s) if _fn_s+_tn_s > 0 else 0 for _fn_s, _tn_s in zip(fn_s, tn_s)]

        false_discovery_rate_c = [_fp_c/(_fp_c+_tp_c) if _fp_c+_tp_c > 0 else 0 for _fp_c, _tp_c in zip(fp_c, tp_c) ]
        false_omission_rate_c = [_fn_c/(_fn_c+_tn_c) if _fn_c+_tn_c > 0 else 0 for _fn_c, _tn_c in zip(fn_c, tn_c)]

        global_comparison = {
            "single": {
                "num_pags": len(results_s),
                "shd": shd_s,
                "tp": tp_s,
                "tn": tn_s,
                "fp": fp_s,
                "fn": fn_s,
                "fdr": false_discovery_rate_s,
                "for": false_omission_rate_s
            },
            "coop": {
                "num_pags": len(results_c),
                "shd": shd_c,
                "tp": tp_c,
                "tn": tn_c,
                "fp": fp_c,
                "fn": fn_c,
                "fdr": false_discovery_rate_c,
                "for": false_omission_rate_c
            }
        }
    else:
        global_comparison = None

    shd_sl, tp_sl, tn_sl, fp_sl, fn_sl, other_sl, correct_edges_sl = results_s_local
    shd_cl, tp_cl, tn_cl, fp_cl, fn_cl, other_cl, correct_edges_cl = results_c_local

    false_discovery_rate_sl = fp_sl/(fp_sl+tp_sl) if fp_sl+tp_sl > 0 else 0
    false_omission_rate_sl = fn_sl/(fn_sl+tn_sl) if fn_sl+tn_sl > 0 else 0

    false_discovery_rate_cl = fp_cl/(fp_cl+tp_cl) if fp_cl+tp_cl > 0 else 0
    false_omission_rate_cl = fn_cl/(fn_cl+tn_cl) if fn_cl+tn_cl > 0 else 0

    result = {
        "alpha": ALPHA,
        "num_samples": NUM_SAMPLES,
        "single_client_data_fraction": CLIENT_A_DATA_FRACTION,
        "global": global_comparison,
        "local": {
            "single": {
                "shd": shd_sl,
                "tp": tp_sl,
                "tn": tn_sl,
                "fp": fp_sl,
                "fn": fn_sl,
                "fdr": false_discovery_rate_sl,
                "for": false_omission_rate_sl
            },
            "coop": {
                "shd": shd_cl,
                "tp": tp_cl,
                "tn": tn_cl,
                "fp": fp_cl,
                "fn": fn_cl,
                "fdr": false_discovery_rate_cl,
                "for": false_omission_rate_cl
            }
        }
    }

    with open(Path(target_dir) / target_file, "a") as f:
        f.write(json.dumps(result) + '\n')

ALPHA = 0.05
NUM_TESTS = 20

#test_setups = test_setups[:1]

test_setups *= NUM_TESTS

for test_setup in tqdm(test_setups, desc='Running Simulation'):
    (true_pag, all_labels), (server_single, server_coop) = setup_servers(test_setup)

    results_single = server_single.run()
    all_labels_single = sorted(list(server_single.schema.keys()))
    single_labels_coop = {id: sorted(list(schema.keys())) for id, schema in server_single.client_schemas.items()}
    df_single = server_results_to_dataframe(all_labels_single, results_single)

    results_coop = server_coop.run()
    all_labels_coop = sorted(list(server_coop.schema.keys()))
    client_labels_coop = {id: sorted(list(schema.keys())) for id, schema in server_coop.client_schemas.items()}
    df_coop = server_results_to_dataframe(all_labels_coop, results_coop)

    pag_list_single, pag_labels_single, pag_list_single_per_client, pag_labels_single_per_client = run_riod(df_single, all_labels_single, single_labels_coop, ALPHA)
    pag_list_coop, pag_labels_coop, pag_list_coop_per_client, pag_labels_coop_per_client = run_riod(df_coop, all_labels_coop, client_labels_coop, ALPHA)

    client_id, pred_pag_single = list(pag_list_single_per_client.items())[0]
    pred_pag_coop = pag_list_coop_per_client[client_id]

    # Only compare local graphs
    metrics_single_local = evaluate_prediction(true_pag, pred_pag_single, all_labels, all_labels_single)
    metrics_coop_local = evaluate_prediction(true_pag, pred_pag_coop, all_labels, all_labels_single)


    # Compare graph subset
    pag_labels_single = sorted(pag_labels_single)
    pag_labels_coop = sorted(pag_labels_coop)

    metrics_single = []
    for pred_pag_single, _pag_labels_single in zip(pag_list_single, pag_labels_single):
        # _pag_labels_single should always equal labels_single
        metrics = evaluate_prediction(true_pag, pred_pag_single, all_labels, all_labels_single)
        metrics_single.append(metrics)

    metrics_coop = []
    for pred_pag_coop, _pag_labels_coop in zip(pag_list_coop, pag_labels_coop):
        # also use reduced vertice set for better comparison (instead of _pag_labels_coop)
        metrics = evaluate_prediction(true_pag, pred_pag_coop, all_labels, all_labels_single)
        metrics_coop.append(metrics)

    # log metrics
    log_results('./experiments/simulation/s1', 'data.ndjson', metrics_single, metrics_coop, metrics_single_local, metrics_coop_local)
