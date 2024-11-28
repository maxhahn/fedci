import polars as pl
import hvplot.polars
import hvplot
import panel as pn
import glob

# Load data
path =  './experiments/r5/*ndjson'
try:
    df = pl.read_ndjson(path)
except:
    import json
    ds = []
    len_orig = 0
    for file in glob.glob(path):
        with open(file, 'r') as f:
            _ds = f.readlines()
        len_orig += len(_ds)
        _ds = [d for d in _ds if 'NaN' not in d]
        _ds = [json.loads(d) for d in _ds]
        ds += _ds
    print(f'{len_orig - len(ds)} entries with at least one NaN value')
    df = pl.from_dicts(ds)

df = df.with_columns(experiment_type=pl.col('name').str.slice(0,3))
#print(df['experiment_type'].value_counts())
pl.Config.set_tbl_rows(20)
print(df.group_by('name', 'num_clients', 'num_samples').len().sort('len', 'num_samples', 'num_clients'))
