import polars as pl
import polars.selectors as cs

# Load the JSON file using polars
def load_and_prepare_data(json_file):
    # Load the JSON file as a Polars DataFrame
    df = pl.read_ndjson(json_file)
    df = df.filter((pl.col('fedci').list.len() > 1) & (pl.col('pvalagg').list.len() > 1))

    # Explode the `fedci` and `pvalagg` columns to compare them
    #df = df.explode(['fedci', 'pvalagg'])
    df = df.explode(['fedci']).explode(['pvalagg'])

    # Expand the metrics in `fedci` and `pvalagg` into separate columns
    df = df.with_columns(
        pl.col('fedci').struct.unnest().name.prefix("fedci_"),
        pl.col('pvalagg').struct.unnest().name.prefix("pvalagg_")
    )

    # Drop the original exploded columns
    df = df.drop(['fedci', 'pvalagg'])

    return df

# Analysis of fedci vs. pvalagg
def analyze_fedci_vs_pvalagg(df):
    # Compare key metrics between fedci and pvalagg
    metrics = [col for col in df.columns if col.startswith("fedci_")]

    # Calculate differences between fedci and pvalagg metrics
    for metric in metrics:
        metric_name = metric.replace("fedci_", "")
        df = df.with_columns(
            (pl.col(f"fedci_{metric_name}") - pl.col(f"pvalagg_{metric_name}"))
            .alias(f"diff_{metric_name}")
        )

    # Example analysis: Mean difference of metrics
    mean_diffs = df.select(
        [pl.col(f"diff_{metric.replace('fedci_', '')}").mean().alias(f"mean_diff_{metric.replace('fedci_', '')}") for metric in metrics]
    )

    return df, mean_diffs

pl.Config.set_tbl_rows(100)

# Filepath to the JSON file
json_file = "experiments/simulation/pvalagg_vs_fedci/*.ndjson"
df = pl.read_ndjson(json_file)

#df = df.filter(pl.col('single_client_data_fraction') == 0.5)
#print(len(df))
#df.write_ndjson(json_file)

grouping_keys = ['name', 'num_samples', 'num_clients', 'alpha']
print(f'=== DF LENGTH: {len(df)} ===')

print('=== SAMPLES PER GROUP ===')
print(df.group_by(grouping_keys).agg(pl.len()))


print('=== NO VALID PREDICTIONS ===')
print(df.with_columns(no_pags=pl.col('metrics').list.len() == 0).group_by('name').agg(pl.col('no_pags').mean()))
#print(df.group_by('name').agg(nulls=pl.col('metric_SHD').null_count()/pl.len()))

#df1 = df.drop('fedci').rename({'pvalagg': 'metrics'}).with_columns(name=pl.lit('p_val_agg'))
#df2 = df.drop('pvalagg').rename({'fedci': 'metrics'}).with_columns(name=pl.lit('fedci'))
#pl.concat([df1,df2]).write_ndjson("experiments/simulation/s3/new_data.ndjson")

# df = df.drop('alternative_metrics').explode('metrics')
# df = df.with_columns(
#     pl.col('metrics').struct.unnest().name.prefix("metric_")
# )

df = df.drop('metrics').explode('alternative_metrics')
df = df.with_columns(
    pl.col('alternative_metrics').struct.unnest().name.prefix("metric_")
)
df = df.with_columns(
    pl.col('metric_Edge_Type_Metrics').struct.unnest().name.prefix("metric_")
).drop('metric_Edge_Type_Metrics')
df = df.with_columns(
    pl.col('metric_Dot Head').struct.unnest().name.prefix("metric_dot_head_"),
    pl.col('metric_Arrow Head').struct.unnest().name.prefix("metric_arrow_head_"),
    pl.col('metric_Tail').struct.unnest().name.prefix("metric_tail_")
).drop('metric_Dot Head', 'metric_Arrow Head', 'metric_Tail')
#print(df)

#print('=== NULLS VALUES ===')
#print(df.group_by('name').agg(nulls=pl.col('metric_SHD').null_count()/pl.len()))
#print(df.group_by(grouping_keys).agg(nulls=pl.col('metric_SHD').null_count()/pl.len()))

cols = cs.ends_with('_SHD') | cs.ends_with('_FOR') | cs.ends_with('_FDR')
cols2 = cs.ends_with('_Recall') | cs.ends_with('_Precision') | cs.ends_with('_F1 Score')
cols3 = cs.starts_with('metric_')


df_agg = df.group_by(grouping_keys).agg(pl.len(), cols3.drop_nulls().mean()).sort(grouping_keys[::-1])
#df_agg = df.group_by(grouping_keys).agg(pl.len(), cols2.mean()).sort(grouping_keys[::-1])
print('=== OVERVIEW ===')
print(df_agg)

import hvplot.polars
import hvplot

# plot = df.hvplot.scatter(
#     x='num_samples',
#     y='metric_SHD',
#     alpha=0.7,
#     ylim=(-0.1,1.1),
#     #xlim=(-0.1,1.1),
#     height=400,
#     width=400,
#     row='num_clients',
#     col='name',
#     #groupby=['num_clients', 'num_samples'],
#     subplots=True,
#     #widget_location='bottom'
#     )

# #metric_FDR
# #metric_FOR
# #metric_SHD
# plot = df.hvplot.box(
#     by='name',
#     y='metric_FOR',
#     #alpha=0.7,
#     #ylim=(-0.1,1.1),
#     #xlim=(-0.1,1.1),
#     height=400,
#     width=400,
#     row='num_clients',
#     col='num_samples',
#     #groupby=['num_clients', 'num_samples'],
#     subplots=True,
#     #widget_location='bottom'
#     )

# hvplot.save(plot, 'images/test.html')
