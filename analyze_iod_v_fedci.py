import polars as pl
import polars.selectors as cs

pl.Config.set_tbl_rows(100)


json_file = "experiments/simulation/pvalagg_vs_fedci/*.ndjson"
df = pl.read_ndjson(json_file)

grouping_keys = ['name', 'num_samples', 'num_clients', 'alpha']

df = df.with_columns(has_prediction_fedci=pl.col('metrics_fedci').list.len() > 0)
df = df.with_columns(has_prediction_pvalagg=pl.col('metrics_pvalagg').list.len() > 0)

print(df.select('no_pred_fedci', 'no_pred_pvalagg').mean())
print(df.group_by('no_pred_fedci', 'no_pred_pvalagg').len())

print(df.columns)

df = df.with_row_index()

df_fedci = df.explode('metrics_fedci')
df_fedci = df_fedci.with_columns(pl.col('metrics_fedci').struct.unnest().name.prefix('metric_')).drop('metrics_fedci')
df_fedci = df_fedci.group_by('index').agg(
    cs.starts_with('metric_').mean().name.suffix('_mean'),
    cs.starts_with('metric_').min().name.suffix('_min'),
    cs.starts_with('metric_').max().name.suffix('_max'),
    len=pl.len()
)

df_pvalagg = df.explode('metrics_pvalagg')
df_pvalagg = df_pvalagg.with_columns(pl.col('metrics_pvalagg').struct.unnest().name.prefix('metric_')).drop('metrics_pvalagg')
df_pvalagg = df_pvalagg.group_by('index').agg(
    cs.starts_with('metric_').mean().name.suffix('_mean'),
    cs.starts_with('metric_').min().name.suffix('_min'),
    cs.starts_with('metric_').max().name.suffix('_max'),
    len=pl.len()
)
print(df_fedci.columns)
df_combined = pl.concat([df_fedci, df_pvalagg])
df = df_combined.join(df.select(['index'] + grouping_keys), on='index', how='inner')

#df = df.join(df_fedci, on=['index'], how='left')
#df = df.join(df_pvalagg, on=['index'], how='left')

#df = df.with_columns()
#print(df)


import hvplot.polars
import hvplot

#metric_FDR
#metric_FOR
#metric_SHD
plot = df.sort(
    'num_clients', 'num_samples'
).hvplot.box(
    by='name',
    y='metric_SHD_mean',
    #alpha=0.7,
    #ylim=(-0.1,1.1),
    #xlim=(-0.1,1.1),
    height=400,
    width=400,
    row='num_clients',
    col='num_samples',
    #groupby=['num_clients', 'num_samples'],
    subplots=True,
    #widget_location='bottom'
    )

hvplot.save(plot, 'images/test.html')
