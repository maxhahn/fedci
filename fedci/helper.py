import math
import os
import copy
import fcntl
import json
from pathlib import Path
import numpy as np
import random

from dgp import NodeCollection

from .server import Server
from .client import Client
from .evaluation import get_symmetric_likelihood_tests, get_riod_tests, compare_tests_to_truth
from .env import DEBUG, EXPAND_ORDINALS, LOG_R, LR, RIDGE

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

def run_configured_test(config, seed=None):
    node_collection, num_samples, num_clients, target_directory, target_file, test_targets = config
    if not os.path.exists(target_directory) and not (DEBUG >= 1):
        os.makedirs(target_directory, exist_ok=True)
    target_file = f'{os.getpid()}-{target_file}'
    return run_test(dgp_nodes=node_collection,
                    num_samples=num_samples,
                    num_clients=num_clients,
                    target_directory=target_directory,
                    target_file=target_file,
                    test_targets=test_targets,
                    seed=seed
                    )

def run_test(dgp_nodes: NodeCollection,
             num_samples,
             num_clients,
             target_directory,
             target_file,
             max_regressors=None,
             test_targets=None,
             seed=None
             ):
    if seed is None:
        seed = random.randrange(2**30)
    if DEBUG >= 1: print(f'Current seed: {seed}')

    random.seed(2*seed)
    np.random.seed(seed)

    dgp_nodes = copy.deepcopy(dgp_nodes)

    dgp_nodes.reset()
    data = dgp_nodes.get(num_samples)

    #data.write_parquet('error-data-01.parquet')

    return run_test_on_data(data,
                            dgp_nodes.name,
                            num_clients,
                            target_directory,
                            target_file,
                            max_regressors,
                            seed=seed,
                            test_targets=test_targets
                            )

def run_test_on_data(data,
                     data_name,
                     num_clients,
                     target_directory,
                     target_file,
                     max_regressors=None,
                     seed=None,
                     test_targets=None
                     ):
    if DEBUG >= 1:
        print("*** Data schema")
        for col, dtype in sorted(data.schema.items(), key=lambda x: x[0]):
            print(f"{col} - {dtype}")

    if LOG_R == 0:
        cb.consolewrite_print = lambda x: None
        cb.consolewrite_warnerror = lambda x: None

    clients = {i:Client(chunk) for i, chunk in enumerate(partition_dataframe(data, num_clients))}
    server = Server(
        clients,
        max_regressors=max_regressors,
        test_targets=test_targets,
        max_iterations=25
        )

    server.run()

    likelihood_ratio_tests = get_symmetric_likelihood_tests(server.get_tests(), test_targets=test_targets)
    baseline_tests = get_riod_tests(data, max_regressors=max_regressors, test_targets=test_targets)
    predicted_p_values, baseline_p_values = compare_tests_to_truth(likelihood_ratio_tests, baseline_tests, test_targets)

    #assert all([abs(a-b) < 0.3 for a,b in zip(predicted_p_values, baseline_p_values)])

    result = {
        'name': data_name,
        'num_clients': num_clients,
        'num_samples': len(data),
        'max_regressors': max_regressors,
        'expanded_ordinals': True if EXPAND_ORDINALS == 1 else False,
        'lr': LR,
        'ridge': RIDGE,
        'seed': seed,
        'predicted_p_values': predicted_p_values,
        'baseline_p_values': baseline_p_values,
        'test_targets': test_targets
    }

    if DEBUG == 0:
        write_result(result, target_directory, target_file)

    return list(zip(sorted(likelihood_ratio_tests), sorted(baseline_tests)))
