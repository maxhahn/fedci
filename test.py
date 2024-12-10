import dgp
from fedci import run_test, run_configured_test, run_test_on_data
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import itertools
import random

# Run with:
# EXPAND_ORDINALS=1 python3 test.py

# Setup Data
# ## L-B CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.BinaryNode])
nc911 = dgp.NodeCollection('C-B Unc. Indep. : X Y', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.BinaryNode], parents=[node1])
nc912 = dgp.NodeCollection('C-B Unc. Dep. : X -> Y', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.BinaryNode])
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc913 = dgp.NodeCollection('C-B Con. Dep. : X -> Z <- Y', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.BinaryNode])
nc914 = dgp.NodeCollection('C-B Con. Indep. : X <- Z -> Y', [node1, node2, node3])
# Unc. Conf. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.BinaryNode])
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc915 = dgp.NodeCollection('C-B Unc. Conf. Indep. : X (-> Z <-) Y', [node1, node2, node3], drop_vars=['Z'])
# Unc. Conf. Dep. Case
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.BinaryNode])
nc916 = dgp.NodeCollection('C-B Unc. Conf. Dep. : X (<- Z ->) Y', [node1, node2, node3], drop_vars=['Z'])

## L-M CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.CategoricalNode], min_categories=3)
nc921 = dgp.NodeCollection('C-M Unc. Indep.', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.CategoricalNode], parents=[node1], min_categories=3)
nc922 = dgp.NodeCollection('C-M Unc. Dep.', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.CategoricalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc923 = dgp.NodeCollection('C-M Con. Dep.', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.CategoricalNode], min_categories=3)
nc924 = dgp.NodeCollection('C-M Con. Indep.', [node1, node2, node3])
# Unc. Conf. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.CategoricalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc925 = dgp.NodeCollection('C-M Unc. Conf. Indep. : X (-> Z <-) Y', [node1, node2, node3], drop_vars=['Z'])
# Unc. Conf. Dep. Case
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.CategoricalNode], min_categories=3)
nc926 = dgp.NodeCollection('C-M Unc. Conf. Dep. : X (<- Z ->) Y', [node1, node2, node3], drop_vars=['Z'])

## L-O CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc931 = dgp.NodeCollection('C-O Unc. Indep.', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], parents=[node1], min_categories=3)
nc932 = dgp.NodeCollection('C-O Unc. Dep.', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc933 = dgp.NodeCollection('C-O Con. Dep.', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc934 = dgp.NodeCollection('C-O Con. Indep.', [node1, node2, node3])
# Unc. Conf. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc935 = dgp.NodeCollection('C-O Unc. Conf. Indep. : X (-> Z <-) Y', [node1, node2, node3], drop_vars=['Z'])
# Unc. Conf. Dep. Case
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc936 = dgp.NodeCollection('C-O Unc. Conf. Dep. : X (<- Z ->) Y', [node1, node2, node3], drop_vars=['Z'])

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
# Unc. Conf. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.BinaryNode])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc945 = dgp.NodeCollection('B-O Unc. Conf. Indep. : X (-> Z <-) Y', [node1, node2, node3], drop_vars=['Z'])
# Unc. Conf. Dep. Case
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.BinaryNode])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc946 = dgp.NodeCollection('B-O Unc. Conf. Dep. : X (<- Z ->) Y', [node1, node2, node3], drop_vars=['Z'])

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
# Unc. Conf. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.CategoricalNode], min_categories=3)
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc955 = dgp.NodeCollection('M-O Unc. Conf. Indep. : X (-> Z <-) Y', [node1, node2, node3], drop_vars=['Z'])
# Unc. Conf. Dep. Case
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.CategoricalNode], min_categories=3)
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc956 = dgp.NodeCollection('M-O Unc. Conf. Dep. : X (<- Z ->) Y', [node1, node2, node3], drop_vars=['Z'])

## L-L CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.Node])
nc961 = dgp.NodeCollection('C-C Unc. Indep.', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.Node], parents=[node1])
nc962 = dgp.NodeCollection('C-C Unc. Dep.', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc963 = dgp.NodeCollection('C-C Con. Dep.', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.Node])
nc964 = dgp.NodeCollection('C-C Con. Indep.', [node1, node2, node3])
# Unc. Conf. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc965 = dgp.NodeCollection('C-C Unc. Conf. Indep. : X (-> Z <-) Y', [node1, node2, node3], drop_vars=['Z'])
# Unc. Conf. Dep. Case
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.Node])
nc966 = dgp.NodeCollection('C-C Unc. Conf. Dep. : X (<- Z ->) Y', [node1, node2, node3], drop_vars=['Z'])

# Setup Configs
node_collections = [
    #nc911, nc912, nc913, nc914, nc915, nc916,
    #nc921, nc922, nc923, nc924, nc925, nc926,
    #nc931, nc932, nc933, nc934, nc935, nc936,
    #nc941, nc942, nc943, nc944, nc945, nc946,
    #nc951, nc952, nc953, nc954, nc955, nc956,
    nc953
    #nc961, nc962, nc963, nc964, nc965, nc966,
]

num_samples = [
    #100,
    100, #300, 400,
    #500, #600, 700, 800,
    #750,
    #900,
    #1000,
    #1250,
    #1500,
    #1750,
    #2000,
    #2500,
    #3000
]
num_clients = [
    1, #3, 5
]

file_info = ('./experiments/r5', 'tests.ndjson')

configurations = list(itertools.product(node_collections, num_samples, num_clients))
configurations = [c + file_info for c in configurations]

num_runs = 1

configurations *= num_runs

bad_seeds = []
#(p1: 0.526709 and p2: 0.6335279)
#run_configured_test(configurations[0] + ([('X', 'Y', ('Z',))],), seed=51)
#for i in range(1000):
#    run_configured_test(configurations[0] + ([('X', 'Y', ('Z',))],), seed=i+40)
import polars as pl
data = pl.read_parquet('error-data-73.parquet')

#data.with_columns(pl.col('X') == '2').write_parquet('error-data-73-booled.parquet')
pl.Config.set_tbl_rows(30)
print(data.group_by('X', 'Y').len().sort('X', 'Y'))
#print(data.cast(pl.Float32).corr())

run_test_on_data(data,
                'test',
                1,
                '',
                '',
                None,
                seed=None,
                test_targets=[('X', 'Y', ('Z', ))]
                )
