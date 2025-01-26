# Load PAGs
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
import fcntl

import dgp
import fedci

# supress R log
import rpy2.rinterface_lib.callbacks as cb
cb.consolewrite_print = lambda x: None
cb.consolewrite_warnerror = lambda x: None

#ro.r['source']('./load_pags.r')
#load_pags = ro.globalenv['load_pags']

# 1. removed R multiprocessing (testing tn)
# 2. put rpy2 source file open into mp function
# 3. from rpy2.rinterface_lib import openrlib
# with openrlib.rlock:
#     # Your R function call here
#     pass

# load local-ci script
ro.r['source']('./temp.r')
# load function from R script
load_pags = ro.globalenv['load_pags']
run_ci_test_f = ro.globalenv['run_ci_test']
aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']
iod_on_ci_data_f = ro.globalenv['iod_on_ci_data']

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

def get_data(test_setup, num_samples, num_clients):
    def split_dataframe(df, n):
        if n <= 0:
            raise ValueError("The number of splits 'n' must be greater than 0.")

        # Determine the size of each split
        num_rows = len(df)
        split_size = num_rows // n
        remainder = num_rows % n

        # Create the splits
        splits = []
        start = 0
        for i in range(n):
            # Account for the remainder rows
            extra_row = 1 if i < remainder else 0
            end = start + split_size + extra_row
            splits.append(df[start:end])
            start = end
        return splits

    pag = test_setup[0]
    nc = pag_to_node_collection(pag)

    data = nc.get(num_samples)

    cols = data.columns
    cols_c1 = test_setup[1][0]
    cols_c2 = test_setup[1][1]
    cols_cx = [sorted(cols, key=lambda k: random.random())[:-1] for _ in range(num_clients-2)]

    client_data = [df.select(c) for df, c in zip(split_dataframe(data, num_clients), [cols_c1, cols_c2] + cols_cx)]

    return (pag, sorted(data.columns)), client_data

def setup_server(client_data):
    # Create Clients
    clients = [fedci.Client(d) for d in client_data]

    # Create Server
    server = fedci.Server(
        {
            str(i): c for i, c in enumerate(clients)
        }
    )

    return server

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
# Run fedci
#server.run()

# Run MXM local-ci.r per Client

def mxm_ci_test(df):
    df = df.to_pandas()
    print(' - launch R conversion context')
    with (ro.default_converter + pandas2ri.converter).context():
        # # load local-ci script
        # ro.r['source']('./local-ci.r')
        # # load function from R script
        # run_ci_test_f = ro.globalenv['run_ci_test']
        print(' - convert df')
        #converting it into r object for passing into r function
        df_r = ro.conversion.get_conversion().py2rpy(df)
        print(' - launch R')
        #Invoking the R function and getting the result
        result = run_ci_test_f(df_r, 999, "./examples/", 'dummy')
        print(' - completed R')
        #Converting it back to a pandas dataframe.
        df_pvals = ro.conversion.get_conversion().rpy2py(result['citestResults'])
        print(' - completed df conversion')
        labels = list(result['labels'])
    return df_pvals, labels

def run_pval_agg_iod(users, dfs, client_labels, alpha):
    #ro.r['source']('./aggregation.r')
    #aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']

    with (ro.default_converter + pandas2ri.converter).context():
        lvs = []
        r_dfs = [ro.conversion.get_conversion().py2rpy(df) for df in dfs]
        #r_dfs = ro.ListVector(r_dfs)
        label_list = [ro.StrVector(v) for v in client_labels]

        result = aggregate_ci_results_f(label_list, r_dfs, alpha)

        g_pag_list = [x[1].tolist() for x in result['G_PAG_List'].items()]
        g_pag_labels = [list(x[1]) for x in result['G_PAG_Label_List'].items()]
        gi_pag_list = [x[1].tolist() for x in result['Gi_PAG_list'].items()]
        gi_pag_labels = [list(x[1]) for x in result['Gi_PAG_Label_List'].items()]
        return g_pag_list, g_pag_labels,  {u:r for u,r in zip(users, gi_pag_list)}, {u:l for u,l in zip(users, gi_pag_labels)}

def run_riod(df, labels, client_labels, alpha):
    # ro.r['source']('./aggregation.r')
    # iod_on_ci_data_f = ro.globalenv['iod_on_ci_data']
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
        gi_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]
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

def log_results(target_dir, target_file, name, metrics, metrics2, alpha, num_samples, num_clients):
    result = {
        "name": name,
        "alpha": alpha,
        "num_samples": num_samples,
        "num_clients": num_clients,
        "metrics": metrics,
        "alternative_metrics": metrics2
    }

    with open(Path(target_dir) / target_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(result) + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)



### sklearn metrics
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

def calculate_pag_metrics(true_pag, predicted_pags, true_labels, predicted_labels_list):
    metrics_list = []

    def adjacency_matrix_to_edges(matrix, labels):
        """ Convert adjacency matrix to edge list with label ordering """
        n = len(labels)
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if matrix[i, j] > 0:  # if there is a directed edge
                    edges.append((labels[i], labels[j]))
                if matrix[j, i] > 0:  # for undirected edges
                    edges.append((labels[j], labels[i]))
        return set(edges)

    def structural_hamming_distance(edges_true, edges_pred):
        """ Structural Hamming Distance """
        return len(edges_true.symmetric_difference(edges_pred)) / len(edges_true.union(edges_pred))

    def false_discovery_rate(edges_true, edges_pred):
        """ False Discovery Rate """
        fp = len(edges_pred - edges_true)
        tp = len(edges_true & edges_pred)
        return fp / (fp + tp) if (fp + tp) > 0 else 0

    def false_omission_rate(edges_true, edges_pred):
        """ False Omission Rate """
        fn = len(edges_true - edges_pred)
        tn = len(edges_true.union(edges_pred)) - len(edges_true & edges_pred)
        return fn / (fn + tn) if (fn + tn) > 0 else 0

    for pag, predicted_labels in zip(predicted_pags, predicted_labels_list):
        edges_true = adjacency_matrix_to_edges(true_pag, true_labels)
        edges_pred = adjacency_matrix_to_edges(pag, predicted_labels)

        shd = structural_hamming_distance(edges_true, edges_pred)
        fdr = false_discovery_rate(edges_true, edges_pred)
        for_ = false_omission_rate(edges_true, edges_pred)

        # Calculating precision, recall, and F1-score
        true_positive = len(edges_true & edges_pred)
        false_positive = len(edges_pred - edges_true)
        false_negative = len(edges_true - edges_pred)
        true_negative = len(edges_true.union(edges_pred)) - true_positive

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            "SHD": shd,
            "FDR": fdr,
            "FOR": for_,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_score
        }

        metrics_list.append(metrics)

    return metrics_list

## MORE METRIC ATTEMPTS
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def reorder_adjacency_matrix(adj_matrix, variables, target_variables):
    """
    Reorder adjacency matrix based on target variable order.
    """
    index_map = {var: idx for idx, var in enumerate(variables)}
    reorder_indices = [index_map[var] for var in target_variables]
    reordered_matrix = adj_matrix[np.ix_(reorder_indices, reorder_indices)]
    return reordered_matrix

def _compare_pags(matrix1, variables1, matrix2, variables2):
    """
    Compare two PAGs and compute SHD, precision, recall, and F1 score.
    """
    # Convert to numpy arrays
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    # Reorder matrices to match variable order
    matrix1 = reorder_adjacency_matrix(matrix1, variables1, variables2)

    # Compute SHD
    shd = np.sum(matrix1 != matrix2).item()

    # Flatten and compare edges of different types
    flat1 = matrix1.flatten()
    flat2 = matrix2.flatten()

    metrics = {}
    for edge_type, label in [(1, "Dot Head"), (2, "Arrow Head"), (3, "Tail")]:
        mask = (flat1 == edge_type) | (flat2 == edge_type)  # Only compare relevant edges
        y_true = flat1[mask] == edge_type
        y_pred = flat2[mask] == edge_type

        precision = precision_score(y_true, y_pred, zero_division=np.nan)
        recall = recall_score(y_true, y_pred, zero_division=np.nan)
        f1 = f1_score(y_true, y_pred, zero_division=np.nan)

        metrics[label] = {
            "Precision": precision if not np.isnan(precision) else None,
            "Recall": recall if not np.isnan(recall) else None,
            "F1_Score": f1 if not np.isnan(f1) else None,
        }

    return {
        "SHD": shd,
        "Edge_Type_Metrics": metrics,
    }

def compare_pags(true_pag, true_labels, pred_pags, pred_labels):
    metrics = []
    for pred_pag, _pred_labels in zip(pred_pags, pred_labels):
        metric = _compare_pags(true_pag, true_labels, pred_pag, _pred_labels)
        metrics.append(metric)
    return metrics

test_setups = list(zip(truePAGs, subsetsList))

NUM_TESTS = 1
ALPHA = 0.05

#test_setups = test_setups[5:10]
data_dir = './experiments/simulation/pvalagg_vs_fedci'
data_file_pattern = '{}-{}-{}.ndjson'

import datetime
import polars as pl

now = int(datetime.datetime.utcnow().timestamp()*1e3)
data_file_pattern = str(now) + '-' + data_file_pattern

def run_comparison(setup):
    idx, data_dir, data_file_pattern, test_setup, num_samples, num_clients = setup
    data_file = data_file_pattern.format(idx, num_samples, num_clients)
    (true_pag, all_labels), client_data = get_data(test_setup, num_samples, num_clients)

    #x = [d.with_columns(client_id=pl.lit(i)) for i, d in enumerate(client_data)]
    #pl.concat(x, how='diagonal').write_parquet('test_data.parquet')

    # x = pl.read_parquet('test_data.parquet')
    # client_data = x.partition_by('client_id')
    # #df.select(pl.all().is_null().all()).unpivot().filter(pl.col('value') == False)['variable'].to_list()
    # client_data = [
    #     d.select(d.drop('client_id').select(pl.all().is_null().all()).unpivot().filter(pl.col('value') == False)['variable'].to_list())
    #     for d in client_data
    # ]
    # from functools import reduce
    # #union_result = reduce(lambda x, y: x.union(y), list_of_sets)
    # all_labels = sorted(list(reduce(lambda x, y: x.union(y), [set(d.columns) for d in client_data])))

    #print('start')

    #print('setup fedci')
    server = setup_server(client_data)

    results_fedci = server.run()
    all_labels_fedci = sorted(list(server.schema.keys()))
    client_labels = {id: sorted(list(schema.keys())) for id, schema in server.client_schemas.items()}
    df_fedci = server_results_to_dataframe(all_labels_fedci, results_fedci)

    #print('iod fedci')
    pag_list, pag_labels, _, _ = run_riod(df_fedci, all_labels_fedci, client_labels, ALPHA)

    #print('metrics fedci')
    metrics = calculate_pag_metrics(
        np.array(true_pag),
        [np.array(p) for p in pag_list],
        all_labels,
        pag_labels
    )
    metrics2 = compare_pags(
        true_pag,
        all_labels,
        pag_list,
        pag_labels
    )

    #print('log fedci')
    log_results(data_dir, data_file, 'fedci', metrics, metrics2, ALPHA, num_samples, num_clients)

    ## Run p val agg IOD
    print('setup pvalagg')
    client_ci_info = []
    for i,d in enumerate(client_data):
        print(f'=== {i}')
        d.write_parquet('test_subdata.parquet')
        client_ci_info.append(mxm_ci_test(d))
    #client_ci_info = [mxm_ci_test(d) for d in client_data]
    print('setup done pvalagg')
    #client_A_ci_df, client_A_labels = mxm_ci_test(client_A_data)
    #client_B_ci_df, client_B_labels = mxm_ci_test(client_B_data)
    client_ci_dfs, client_ci_labels = zip(*client_ci_info)

    #print('iod pvalagg')
    pag_list, pag_labels, _, _ = run_pval_agg_iod(
        list(client_labels.keys()),
        client_ci_dfs,
        client_ci_labels,
        ALPHA
    )

    #print('metrics pvalagg')
    metrics = calculate_pag_metrics(
        np.array(true_pag),
        [np.array(p) for p in pag_list],
        all_labels,
        pag_labels
    )
    metrics2 = compare_pags(
        true_pag,
        all_labels,
        pag_list,
        pag_labels
    )

    # TODO: one log for both, to match results on same data

    #print('log pvalagg')
    # log metrics
    log_results(data_dir, data_file, 'p_val_agg', metrics, metrics2, ALPHA, num_samples, num_clients)

num_clients_options = [3,5,10]
num_samples_options = [50,100,250,500]

configurations = list(itertools.product(test_setups, num_samples_options, num_clients_options))

configurations = [(data_dir, data_file_pattern) + c for c in configurations]

configurations = [(i,) + c for i in range(NUM_TESTS) for c in configurations]

from tqdm.contrib.concurrent import process_map

for configuration in tqdm(configurations):
    run_comparison(configuration)

#process_map(run_comparison, configurations, max_workers=10, chunksize=3)

# for _ in range(NUM_TESTS):
#     now = int(datetime.datetime.utcnow().timestamp()*1e3)
#     data_file = data_file_pattern.format(now)
#     for num_clients in [3,5,10]:
#         for num_samples in [50,100,250,500]:
#             for test_setup in tqdm(test_setups, desc='Running Simulation'):
#                 (true_pag, all_labels), client_data = get_data(test_setup, num_samples, num_clients)

#                 x = [d.with_columns(client_id=pl.lit(i)) for i, d in enumerate(client_data)]
#                 pl.concat(x, how='diagonal').write_parquet('test_data.parquet')

#                 # x = pl.read_parquet('test_data.parquet')
#                 # client_data = x.partition_by('client_id')
#                 # #df.select(pl.all().is_null().all()).unpivot().filter(pl.col('value') == False)['variable'].to_list()
#                 # client_data = [
#                 #     d.select(d.drop('client_id').select(pl.all().is_null().all()).unpivot().filter(pl.col('value') == False)['variable'].to_list())
#                 #     for d in client_data
#                 # ]
#                 # print(client_data[0])
#                 # from functools import reduce
#                 # #union_result = reduce(lambda x, y: x.union(y), list_of_sets)
#                 # all_labels = sorted(list(reduce(lambda x, y: x.union(y), [set(d.columns) for d in client_data])))


#                 #print('start')

#                 #print('setup fedci')
#                 server = setup_server(client_data)

#                 results_fedci = server.run()
#                 all_labels_fedci = sorted(list(server.schema.keys()))
#                 client_labels = {id: sorted(list(schema.keys())) for id, schema in server.client_schemas.items()}
#                 df_fedci = server_results_to_dataframe(all_labels_fedci, results_fedci)

#                 #print('iod fedci')
#                 pag_list, pag_labels, _, _ = run_riod(df_fedci, all_labels_fedci, client_labels, ALPHA)

#                 #print('metrics fedci')
#                 metrics = calculate_pag_metrics(
#                     np.array(true_pag),
#                     [np.array(p) for p in pag_list],
#                     all_labels,
#                     pag_labels
#                 )

#                 #print('log fedci')
#                 log_results(data_dir, data_file, 'fedci', metrics, ALPHA, num_samples, num_clients)

#                 ## Run p val agg IOD
#                 #print('setup pvalagg')
#                 client_ci_info = [mxm_ci_test(d) for d in client_data]
#                 #client_A_ci_df, client_A_labels = mxm_ci_test(client_A_data)
#                 #client_B_ci_df, client_B_labels = mxm_ci_test(client_B_data)
#                 client_ci_dfs, client_ci_labels = zip(*client_ci_info)

#                 #print('iod pvalagg')
#                 pag_list, pag_labels, _, _ = run_pval_agg_iod(
#                     list(client_labels.keys()),
#                     client_ci_dfs,
#                     client_ci_labels,
#                     ALPHA
#                 )

#                 #print('metrics pvalagg')
#                 metrics = calculate_pag_metrics(
#                     np.array(true_pag),
#                     [np.array(p) for p in pag_list],
#                     all_labels,
#                     pag_labels
#                 )

#                 #print('log pvalagg')
#                 # log metrics
#                 log_results(data_dir, data_file, 'p_val_agg', metrics, ALPHA, num_samples, num_clients)
