import math
import copy
import fcntl
import json
from pathlib import Path

from dgp import NodeCollection

from .server import Server
from .client import Client
from .evaluation import get_symmetric_likelihood_tests, get_riod_tests, compare_tests_to_truth

import rpy2.rinterface_lib.callbacks as cb

def partition_dataframe(df, n):
    total_rows = len(df)
    partition_size = math.ceil(total_rows / n)
    
    partitions = []
    for i in range(n):
        start_idx = i * partition_size
        end_idx = min((i + 1) * partition_size, total_rows)
        partition = df[start_idx:end_idx]
        partitions.append(partition)
    
    return partitions

def write_result(result, directory, file):
    with open(Path(directory) / file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(result) + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)
        
def run_configured_test(config):
    node_collection, num_samples, num_clients, target_directory, target_file = config
    
    run_test(
        dgp_nodes=node_collection,
        num_samples=num_samples,
        num_clients=num_clients,
        target_directory=target_directory,
        target_file=target_file,
        write_to_disk=True
        )
    
def run_test(dgp_nodes: NodeCollection,
             num_samples,
             num_clients,
             target_directory,
             target_file,
             max_regressors=None,
             suppress_r_output=True,
             write_to_disk=True
             ):
    if suppress_r_output:
        cb.consolewrite_print = lambda x: None
        cb.consolewrite_warnerror = lambda x: None

    dgp_nodes = copy.deepcopy(dgp_nodes)
    dgp_nodes.reset()
    data = dgp_nodes.get(num_samples)
    
    clients = {i:Client(chunk) for i, chunk in enumerate(partition_dataframe(data, num_clients))}
    server = Server(
        clients,
        max_regressors=max_regressors
        )
    
    server.run()
    
    likelihood_ratio_tests = get_symmetric_likelihood_tests(server.get_tests())
    ground_truth_tests = get_riod_tests(data, max_regressors=max_regressors)
    predicted_p_values, true_p_values = compare_tests_to_truth(likelihood_ratio_tests, ground_truth_tests)
    
    result = {
        'name': dgp_nodes.name,
        'num_clients': num_clients,
        'num_samples': num_samples,
        'max_regressors': max_regressors,
        'predicted_p_values': predicted_p_values,
        'true_p_values': true_p_values
    }
    
    if write_to_disk:
        write_result(result, target_directory, target_file)