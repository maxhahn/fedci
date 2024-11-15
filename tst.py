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
    #nc911, nc912, nc913, nc914,
    #nc921, nc922, nc923, nc924,
    #nc931, nc932, nc933, nc934,
    nc941, #nc942, nc943, nc944,
    #nc951, nc952,
    #nc953#, nc954,
]
num_samples = [
    #100, 200, 300, 400,
    500, #600, 700, 800,
    #900,
    #1000,# 1250, 1500,
    #1750, 2000, 2500, 3000
]
num_clients = [
    1, #3, 5
]

file_info = ('./experiments/dummy', 'tests.ndjson')

configurations = list(itertools.product(node_collections, num_samples, num_clients))
configurations = [c + file_info for c in configurations]

num_runs = 1

configurations *= num_runs
# Run tests
#process_map(run_configured_test, configurations, max_workers=5, chunksize=10)
#for i, configuration in enumerate(tqdm(configurations[:1], disable=True)):
#    run_configured_test(configuration, 2)
#for i in range(20):
#    run_configured_test(configurations[0], i)
#run_configured_test(configurations[0], 15)
import polars as pl
df = pl.read_parquet("./error-data-02.parquet")
run_test_on_data(
    df,
    "test-data",
    1,
    "experiments/dummy",
    "test.ndjson"
)

import pandas as pd
import statsmodels.api as sm
def run_mnlogit(df, y_var, x_vars):
    # Prepare X matrix with constant
    X = sm.add_constant(df[x_vars])

    # Prepare y variable
    y = df[y_var]

    # Fit the model
    model = sm.MNLogit(y, X)
    results = model.fit()

    # Get coefficients as DataFrame
    coef_df = pd.DataFrame(results.params)

    return coef_df, results.llf


def run_binlogit(df, y_var, x_vars):

    df = df.sort(x_vars, descending=True)
    if len(x_vars) > 0:
        df_x = df[x_vars].to_dummies(x_vars, separator='_ord_', drop_first=True).cast(pl.Int32).to_pandas()
    else:
        df_x = df.to_pandas()[x_vars]
    df = df.to_pandas()
    # Prepare X matrix with constant
    X = sm.add_constant(df_x)

    # Prepare y variable (assume binary 0/1)
    y = df[y_var].astype(int)

    # Fit the binary logistic regression model
    model = sm.Logit(y, X)
    results = model.fit()

    # Get coefficients as DataFrame
    coef_df = pd.DataFrame(results.params)

    return coef_df, results.llf

print("On Intercept")
#r = run_mnlogit(df.to_pandas(), "X", [])
r = run_binlogit(df, "X", [])
print(r[0])
print("llf", r[1])
print("On Y,1")
#r = run_mnlogit(df.to_pandas(), "X", ["Y"])
r = run_binlogit(df, "X", ["Y"])
print(r[0])
print("llf", r[1])

#from pycit import itest

#pval = itest(df["X"].to_numpy(), df["Y"].to_numpy(), test_args={'statistic': 'ksg_mi', 'n_jobs': 2})
#print(f"Pycit pval A = {pval}")
#pval = itest(df["Y"].to_numpy(), df["X"].to_numpy(), test_args={'statistic': 'ksg_mi', 'n_jobs': 2})
#print(f"Pycit pval B = {pval}")
