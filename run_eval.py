
# %%
import polars as pl
import hvplot.polars
import hvplot
import glob

# %%
# Load data
path =  './experiments/t2/*ndjson'
try:
    df = pl.read_ndjson(path, ignore_errors=True)
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

df = df.with_columns(
    experiment_type=pl.col('name').str.slice(0,3),
    conditioning_type=pl.col('name').str.slice(4)
)

print(df.group_by('experiment_type', 'conditioning_type').agg(pl.len()).sort('len'))
df = df.sort('experiment_type', 'conditioning_type')
df = df.explode('predicted_p_values', 'true_p_values')

# Plot scatter of p values
plot = df.hvplot.scatter(
    x='predicted_p_values',
    y='true_p_values',
    alpha=0.7,
    ylim=(-0.1,1.1),
    xlim=(-0.1,1.1),
    height=400,
    width=400,
    row='experiment_type',
    col='conditioning_type',
    groupby=['num_clients', 'num_samples'],
    subplots=True,
    #widget_location='bottom'
    )

hvplot.save(plot, 'images/p_value_scatter.html')

# Plot correlation of p values
_df = df
df_correlation_fix = _df.with_columns(correct=pl.col('predicted_p_values') == pl.col('true_p_values'))
df_correlation_fix = df_correlation_fix.group_by('name', 'num_clients', 'num_samples').agg(all_corrects=pl.min('correct'))
df_correlation_fix = df_correlation_fix.filter(pl.col('all_corrects')).drop('all_corrects')
df_correlation_fix = df_correlation_fix.with_columns(correlation_fix=pl.lit(1.0))

_df = _df.group_by('name', 'experiment_type', 'conditioning_type', 'num_clients', 'num_samples') \
    .agg(pl.corr('predicted_p_values', 'true_p_values')) \
    .rename({'predicted_p_values': 'p_value_correlation'})

_df = _df.join(df_correlation_fix, on=['name', 'num_clients', 'num_samples'], how='left')
_df = _df.with_columns(pl.col('p_value_correlation').replace_strict({float('NaN'): None}, default=pl.col('p_value_correlation')))
_df = _df.with_columns(pl.coalesce(['p_value_correlation', 'correlation_fix'])).drop('correlation_fix')

assert _df['p_value_correlation'].null_count() == 0, 'NaN in correlations'

plot = _df.sort('num_samples').hvplot.line(x='num_samples',
                                  y='p_value_correlation',
                                  alpha=0.6,
                                  row='experiment_type',
                                  col='conditioning_type',
                                  groupby=['num_clients'],
                                  ylim=(-1.5,1.5),
                                  width=400,
                                  height=400,
                                  subplots=True,
                                  #widget_location='bottom'
                                  )

hvplot.save(plot, 'images/p_value_corr.html')

# Plot accuracy
alpha = 0.05

_df = df
_df = _df.with_columns(
    tp=(pl.col('predicted_p_values') < alpha) & (pl.col('true_p_values') < alpha),
    tn=(pl.col('predicted_p_values') > alpha) & (pl.col('true_p_values') > alpha),
    fp=(pl.col('predicted_p_values') < alpha) & (pl.col('true_p_values') > alpha),
    fn=(pl.col('predicted_p_values') > alpha) & (pl.col('true_p_values') < alpha),
)

_df = _df.group_by('name', 'experiment_type', 'conditioning_type', 'num_clients', 'num_samples').agg((pl.col('tp')+pl.col('tn')).mean().alias('accuracy'))

plot = _df.sort('num_samples').hvplot.line(x='num_samples',
                                  y='accuracy',
                                  alpha=0.6,
                                  row='experiment_type',
                                  col='conditioning_type',
                                  groupby=['num_clients'],
                                  ylim=(-1.5,1.5),
                                  width=400,
                                  height=400,
                                  subplots=True,
                                  #widget_location='bottom'
                                  )
hvplot.save(plot, 'images/p_value_acc.html')
