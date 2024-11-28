
# %%
import polars as pl
import hvplot.polars
import hvplot
import glob

# %%
# Load data
path =  './experiments/r5/*ndjson'
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

df_correlation_fix = df.with_columns(correct=(pl.col('predicted_p_values') - pl.col('true_p_values')).round(8) == 0)
df_correlation_fix = df_correlation_fix.group_by(['name', 'num_clients', 'num_samples']).agg(pl.all('correct'))
df_correlation_fix = df_correlation_fix.filter(pl.col('correct')).drop('correct')
df_correlation_fix = df_correlation_fix.with_columns(correlation_fix=pl.lit(1.0))

df_correlation_fix2 = df.with_columns(
    pl.n_unique('predicted_p_values', 'true_p_values').over('name', 'num_clients', 'num_samples').name.suffix('_nunique')
)
df_correlation_fix2 = df_correlation_fix2.filter((pl.col('predicted_p_values_nunique') == 1) | (pl.col('true_p_values_nunique') == 1))
df_correlation_fix2 = df_correlation_fix2.drop('predicted_p_values_nunique', 'true_p_values_nunique')
df_correlation_fix2 = df_correlation_fix2.group_by('name', 'num_clients', 'num_samples').agg(
    mean_correctness=(pl.col('predicted_p_values')==pl.col('true_p_values')).mean(),
    mean_difference_p_value=(pl.col('predicted_p_values').mean()-pl.col('true_p_values').mean()).abs()
)
df_correlation_fix2 = df_correlation_fix2.with_columns(
    correlation_fix=((pl.col('mean_difference_p_value') < 1e-4) & (pl.col('mean_correctness') > 0.9)).cast(pl.Float64)
)
df_correlation_fix2 = df_correlation_fix2.drop('mean_difference_p_value', 'mean_correctness')

_df = _df.group_by(
    'name', 'experiment_type', 'conditioning_type', 'num_clients', 'num_samples'
).agg(
    p_value_correlation=pl.corr('predicted_p_values', 'true_p_values')
)

_df = _df.with_columns(pl.col('p_value_correlation').fill_nan(None))
_df = _df.join(df_correlation_fix, on=['name', 'num_clients', 'num_samples'], how='left')
_df = _df.with_columns(pl.coalesce(['p_value_correlation', 'correlation_fix'])).drop('correlation_fix')

#dfx = _df.filter(pl.col('p_value_correlation').is_null())

_df = _df.with_columns(pl.col('p_value_correlation').fill_nan(None))
_df = _df.join(df_correlation_fix2, on=['name', 'num_clients', 'num_samples'], how='left')
_df = _df.with_columns(pl.coalesce(['p_value_correlation', 'correlation_fix'])).drop('correlation_fix')

_df = _df.with_columns(pl.col('p_value_correlation').fill_nan(None))

#print(_df.join(dfx, on=['name', 'num_clients', 'num_samples'], how='semi'))
assert _df['p_value_correlation'].null_count() == 0, 'NaN in correlations'

plot = _df.sort('num_samples').hvplot.line(x='num_samples',
                                  y='p_value_correlation',
                                  alpha=0.6,
                                  row='experiment_type',
                                  col='conditioning_type',
                                  groupby=['num_clients'],
                                  ylim=(0.8,1.01),
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
                                  ylim=(0.8,1.01),
                                  width=400,
                                  height=400,
                                  subplots=True,
                                  #widget_location='bottom'
                                  )
hvplot.save(plot, 'images/p_value_acc.html')


# Extension of Jaccard Coefficient for Multi-Sets
# -> Highly unlikely that this is useful, unless there are very very few distinct results
#_df = df
#_df = _df.unpivot(['predicted_p_values', 'true_p_values'], index=['name', 'experiment_type', 'conditioning_type', 'num_clients', 'num_samples'])
#_df = _df.group_by(['name', 'experiment_type', 'conditioning_type', 'num_clients', 'num_samples', 'value', 'variable']).len()
#_df = _df.pivot('variable', values='len').fill_null(pl.lit(0)).drop('value')
#_df = _df.rename({'predicted_p_values': 'predicted', 'true_p_values': 'true'})
#_df = _df.with_columns(
#    min_col=pl.min_horizontal(pl.col('predicted', 'true')),
#    max_col=pl.max_horizontal(pl.col('predicted', 'true'))
#)
#_df = _df.group_by(['name', 'experiment_type', 'conditioning_type', 'num_clients', 'num_samples']).agg(jaccard_correlation=pl.sum('min_col')/pl.sum('max_col'))
#
#plot = _df.sort('num_samples').hvplot.line(x='num_samples',
#                                  y='jaccard_correlation',
#                                  alpha=0.6,
#                                  row='experiment_type',
#                                  col='conditioning_type',
#                                  groupby=['num_clients'],
#                                  ylim=(0.8,1.01),
#                                  width=400,
#                                  height=400,
#                                  subplots=True,
#                                  #widget_location='bottom'
#                                  )
#
#hvplot.save(plot, 'images/p_value_jaccard_overlap.html')
