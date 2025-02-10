
# %%
import polars as pl
import polars.selectors as cs
import hvplot
import hvplot.polars
import glob

import altair as alt

# %%
# Load data
path =  './experiments/fed-v-fisher/*.ndjson'
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


df = df.select('name', 'num_clients', 'num_samples', cs.ends_with('_p_values'))

df = df.sort('name')

df = df.with_columns(
    experiment_type=pl.col('name').str.slice(0,3),
    conditioning_type=pl.col('name').str.slice(4)
)

# quickfix = {
#     'Unc. Indep.' : 'Unc. Indep. : X Y',
#     'Unc. Dep.' : 'Unc. Dep. : X -> Y',
#     'Con. Dep.' : 'Con. Dep. : X -> Z <- Y',
#     'Con. Indep.' : 'Con. Indep. : X <- Z -> Y'
# }

# df = df.with_columns(pl.col('conditioning_type').replace(quickfix))

print(df.group_by('experiment_type', 'conditioning_type').agg(pl.len()).sort('len'))
df = df.sort('experiment_type', 'conditioning_type')
df = df.explode('federated_p_values', 'fisher_p_values', 'baseline_p_values')



df_unpivot = df.unpivot(
    on=['federated_p_values', 'fisher_p_values'],
    index=['name', 'experiment_type', 'conditioning_type', 'num_clients', 'num_samples', 'baseline_p_values'],
    value_name='predicted_p_values',
    variable_name='p_value_type'
)


base = alt.Chart().mark_point(
    opacity=0.5
).encode(
    x='baseline_p_values',
    y='predicted_p_values',
    color='p_value_type'
).properties(
    width=200,
    height=200
).interactive()

chart = alt.vconcat(data=df_unpivot)
for y_encoding in df['experiment_type'].unique().sort().to_list():
    row = alt.hconcat()
    for x_encoding in df['conditioning_type'].unique().sort().to_list():
        row |= base.transform_filter(
            (alt.datum.experiment_type == y_encoding) & (alt.datum.conditioning_type == x_encoding)
        )
    chart &= row
chart.save('images/altair.html')

# Plot correlation of p values
def get_correlation(df, colx, coly):
    _df = df

    df_correlation_fix = df.with_columns(correct=(pl.col(colx) - pl.col(coly)).round(8) == 0)
    df_correlation_fix = df_correlation_fix.group_by(['name', 'num_clients', 'num_samples']).agg(pl.all('correct'))
    df_correlation_fix = df_correlation_fix.filter(pl.col('correct')).drop('correct')
    df_correlation_fix = df_correlation_fix.with_columns(correlation_fix=pl.lit(1.0))

    df_correlation_fix2 = df.with_columns(
        pl.n_unique(colx, coly).over('name', 'num_clients', 'num_samples').name.suffix('_nunique')
    )
    df_correlation_fix2 = df_correlation_fix2.filter((pl.col(f'{colx}_nunique') == 1) | (pl.col(f'{coly}_nunique') == 1))
    df_correlation_fix2 = df_correlation_fix2.drop(f'{colx}_nunique', f'{coly}_nunique')
    df_correlation_fix2 = df_correlation_fix2.group_by('name', 'num_clients', 'num_samples').agg(
        mean_correctness=(pl.col(colx)==pl.col(coly)).mean(),
        mean_difference_p_value=(pl.col(colx).mean()-pl.col(coly).mean()).abs()
    )
    df_correlation_fix2 = df_correlation_fix2.with_columns(
        correlation_fix=((pl.col('mean_difference_p_value') < 1e-4) & (pl.col('mean_correctness') > 0.9)).cast(pl.Float64)
    )
    df_correlation_fix2 = df_correlation_fix2.drop('mean_difference_p_value', 'mean_correctness')

    _df = _df.group_by(
        'name', 'experiment_type', 'conditioning_type', 'num_clients', 'num_samples'
    ).agg(
        p_value_correlation=pl.corr(colx, coly)
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

    return _df

df_fed = get_correlation(df, 'federated_p_values', 'baseline_p_values').rename({'p_value_correlation': 'federated_correlation'})
df_fisher = get_correlation(df, 'fisher_p_values', 'baseline_p_values').rename({'p_value_correlation': 'fisher_correlation'})

identifiers = [
    'name',
    'num_clients',
    'num_samples',
    'experiment_type',
    'conditioning_type'
]
_df = df.select(identifiers).unique().join(
    df_fed, on=identifiers, how='left'
).join(
    df_fisher, on=identifiers, how='left'
)

# df_fed = df_fed.with_columns(technique=pl.lit('federated'))
# df_fisher = df_fisher.with_columns(technique=pl.lit('fisher'))

# _df = pl.concat([df_fed, df_fisher])

plot = _df.sort(
    'num_samples',
    'num_clients',
    'experiment_type',
    'conditioning_type'
).hvplot.line(
    x='num_samples',
    y=['federated_correlation', 'fisher_correlation'],
    alpha=0.8,
    groupby=['num_clients', 'experiment_type', 'conditioning_type'],
    ylim=(-0.01,1.01),
    width=400,
    height=400,
    legend='bottom'
)

hvplot.save(plot, 'images/p_value_corr.html')

# Plot accuracy
alpha = 0.05

_df = df
_df = _df.with_columns(
    federated_tp=(pl.col('federated_p_values') < alpha) & (pl.col('baseline_p_values') < alpha),
    federated_tn=(pl.col('federated_p_values') > alpha) & (pl.col('baseline_p_values') > alpha),
    federated_fp=(pl.col('federated_p_values') < alpha) & (pl.col('baseline_p_values') > alpha),
    federated_fn=(pl.col('federated_p_values') > alpha) & (pl.col('baseline_p_values') < alpha),
)
_df = _df.with_columns(
    fisher_tp=(pl.col('fisher_p_values') < alpha) & (pl.col('baseline_p_values') < alpha),
    fisher_tn=(pl.col('fisher_p_values') > alpha) & (pl.col('baseline_p_values') > alpha),
    fisher_fp=(pl.col('fisher_p_values') < alpha) & (pl.col('baseline_p_values') > alpha),
    fisher_fn=(pl.col('fisher_p_values') > alpha) & (pl.col('baseline_p_values') < alpha),
)

_df = _df.group_by(
    'name', 'experiment_type', 'conditioning_type', 'num_clients', 'num_samples'
).agg(
    (pl.col('federated_tp')+pl.col('federated_tn')).mean().alias('federated_accuracy'),
    (pl.col('fisher_tp')+pl.col('fisher_tn')).mean().alias('fisher_accuracy')
)


plot = _df.sort(
    'num_samples',
    'num_clients',
    'experiment_type',
    'conditioning_type'
).hvplot.line(
    x='num_samples',
    y=['federated_accuracy', 'fisher_accuracy'],
    alpha=0.8,
    groupby=['num_clients', 'experiment_type', 'conditioning_type'],
    ylim=(0.75,1.01),
    width=400,
    height=400,
    legend='bottom'
)

hvplot.save(plot, 'images/p_value_acc.html')

# df_fed = df.with_columns(
#     tp=(pl.col('federated_p_values') < alpha) & (pl.col('baseline_p_values') < alpha),
#     tn=(pl.col('federated_p_values') > alpha) & (pl.col('baseline_p_values') > alpha),
#     fp=(pl.col('federated_p_values') < alpha) & (pl.col('baseline_p_values') > alpha),
#     fn=(pl.col('federated_p_values') > alpha) & (pl.col('baseline_p_values') < alpha),
# )
# df_fisher = df.with_columns(
#     tp=(pl.col('fisher_p_values') < alpha) & (pl.col('baseline_p_values') < alpha),
#     tn=(pl.col('fisher_p_values') > alpha) & (pl.col('baseline_p_values') > alpha),
#     fp=(pl.col('fisher_p_values') < alpha) & (pl.col('baseline_p_values') > alpha),
#     fn=(pl.col('fisher_p_values') > alpha) & (pl.col('baseline_p_values') < alpha),
# )

# df_fed = df_fed.with_columns(technique=pl.lit('federated'))
# df_fisher = df_fisher.with_columns(technique=pl.lit('fisher'))

# _df = pl.concat([df_fed, df_fisher])

# _df = _df.group_by('name', 'experiment_type', 'conditioning_type', 'num_clients', 'num_samples', 'technique').agg((pl.col('tp')+pl.col('tn')).mean().alias('accuracy'))

# # plot = _df.sort('num_samples').hvplot.line(x='num_samples',
# #                                   y='accuracy',
# #                                   alpha=0.6,
# #                                   row='experiment_type',
# #                                   col='conditioning_type',
# #                                   groupby=['num_clients', 'technique'],
# #                                   ylim=(0.75,1.01),
# #                                   width=400,
# #                                   height=400,
# #                                   subplots=True,
# #                                   #widget_location='bottom'
# #                                   )
# plot = _df.sort('num_samples').hvplot.line(x='num_samples',
#                                 y='accuracy',
#                                 alpha=0.6,
#                                 groupby=['num_clients', 'experiment_type', 'conditioning_type'],
#                                 by=['technique'],
#                                 ylim=(0.75,1.01),
#                                 width=400,
#                                 height=400,
#                                 subplots=True,
#                                 #widget_location='bottom'
#                                 )
# hvplot.save(plot, 'images/p_value_acc.html')
