import dgp
from fedci import run_test, run_configured_test, run_test_on_data
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import itertools

# Setup Data
# ## L-B CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.BinaryNode])
nc911 = dgp.NodeCollection('L-B Unc. Indep.', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.BinaryNode], parents=[node1])
nc912 = dgp.NodeCollection('L-B Unc. Dep.', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.BinaryNode])
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc913 = dgp.NodeCollection('L-B Con. Dep.', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.BinaryNode])
nc914 = dgp.NodeCollection('L-B Con. Indep.', [node1, node2, node3])

## L-M CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.CategoricalNode], min_categories=3)
nc921 = dgp.NodeCollection('L-M Unc. Indep.', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.CategoricalNode], parents=[node1], min_categories=3)
nc922 = dgp.NodeCollection('L-M Unc. Dep.', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.CategoricalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc923 = dgp.NodeCollection('L-M Con. Dep.', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.CategoricalNode], min_categories=3)
nc924 = dgp.NodeCollection('L-M Con. Indep.', [node1, node2, node3])

## L-O CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc931 = dgp.NodeCollection('L-O Unc. Indep.', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], parents=[node1], min_categories=3)
nc932 = dgp.NodeCollection('L-O Unc. Dep.', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc933 = dgp.NodeCollection('L-O Con. Dep.', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc934 = dgp.NodeCollection('L-O Con. Indep.', [node1, node2, node3])

## B-O CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.BinaryNode])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc941 = dgp.NodeCollection('B-O Unc. Indep.', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.BinaryNode])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], parents=[node1], min_categories=3)
nc942 = dgp.NodeCollection('B-O Unc. Dep.', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.BinaryNode])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc943 = dgp.NodeCollection('B-O Con. Dep.', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.BinaryNode])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc944 = dgp.NodeCollection('B-O Con. Indep.', [node1, node2, node3])

## M-O CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.CategoricalNode], min_categories=3)
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc951 = dgp.NodeCollection('M-O Unc. Indep.', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.CategoricalNode], min_categories=3)
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], parents=[node1], min_categories=3)
nc952 = dgp.NodeCollection('M-O Unc. Dep.', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.CategoricalNode], min_categories=3)
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc953 = dgp.NodeCollection('M-O Con. Dep.', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.CategoricalNode], min_categories=3)
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc954 = dgp.NodeCollection('M-O Con. Indep.', [node1, node2, node3])

## L-L CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.Node])
nc961 = dgp.NodeCollection('L-L Unc. Indep.', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.Node], parents=[node1])
nc962 = dgp.NodeCollection('L-L Unc. Dep.', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc963 = dgp.NodeCollection('L-L Con. Dep.', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.Node])
nc964 = dgp.NodeCollection('L-L Con. Indep.', [node1, node2, node3])

# Setup Configs
node_collections = [
    nc911, nc912, nc913, nc914,
    nc921, nc922, nc923, nc924,
    nc931, nc932, nc933, nc934,
    nc941, nc942, nc943, nc944,
    nc951, nc952, nc953, nc954,
]
#node_collections = [nc942]
num_samples = [
    100, 200, 300, 400,
    500, #600, 700, 800,
    #900,
    1000,# 1250, 1500,
    #1750, 2000, 2500, 3000
]
num_clients = [
    1, 3#, 5
]

file_info = ('./experiments/expanded_ordinals', 'tests.ndjson')

configurations = list(itertools.product(node_collections, num_samples, num_clients))
configurations = [c + file_info for c in configurations]

num_runs = 25

configurations *= num_runs

# Run tests
process_map(run_configured_test, configurations, max_workers=5, chunksize=10)
#for i, configuration in enumerate(tqdm(configurations)):
#    run_configured_test(configuration)
