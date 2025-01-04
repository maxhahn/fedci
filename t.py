import fedci
import dgp

import random
import math
import numpy as np
import polars as pl

n1 = dgp.GenericNode('X1')
n2 = dgp.GenericNode('X2', parents=[n1])
n3 = dgp.GenericNode('X3', parents=[n2])
n4 = dgp.GenericNode('X4', parents=[n2, n3])
n5 = dgp.GenericNode('X5', parents=[n2, n3])
nc = dgp.NodeCollection('Test Graph', [n1,n2,n3,n4,n5])

NUM_SAMPLES = 1500
NUM_CLIENTS = 3

data = nc.get(NUM_SAMPLES)
split_ratios = [math.exp(random.uniform(0,1)) for _ in range(NUM_CLIENTS)]
split_ratios = [sr/sum(split_ratios) for sr in split_ratios]
split_ratios = np.cumsum(split_ratios)[:-1]

split_ratios = [0.3333, 0.6666, 1]

split_ratios = sorted(split_ratios)
split_offsets = [int(sr*NUM_SAMPLES) for sr in split_ratios]
split_offsets = [0] + split_offsets

client_data = []
for i in range(NUM_CLIENTS):
    split_offset = split_offsets[i]
    split_length = None
    if i+1 != NUM_CLIENTS:
        split_length = split_offsets[i+1] - split_offsets[i]
    client_data.append(data.slice(split_offset, split_length))

# control server
control_server = fedci.Server({'control': fedci.Client(data)})
control_server.run()
control_tests = control_server.get_tests()
control_tests = fedci.get_symmetric_likelihood_tests(control_tests, test_targets=None)

results = []

for _ in range(10):
    # experiment setup
    drop_columns = [[]] + data.columns[:len(client_data)-1]
    #drop_columns = data.columns[:len(client_data)]
    clients = {str(i):fedci.Client(client_data[i].drop(drop_columns[i])) for i in range(NUM_CLIENTS)}

    server = fedci.Server(clients)
    server.run()

    experiment_tests = server.get_tests()
    likelihood_ratio_tests = fedci.get_symmetric_likelihood_tests(server.get_tests(), test_targets=None)
    predicted_p_values, baseline_p_values = fedci.compare_tests_to_truth(likelihood_ratio_tests, control_tests, test_targets=None)

    result = {
        'predicted': predicted_p_values,
        'baseline': baseline_p_values
    }

    results.append(result)

df = pl.from_dicts(results)
df.write_ndjson('./experiments/tmp2/test.ndjson')
