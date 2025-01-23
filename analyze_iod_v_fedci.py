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

# Filepath to the JSON file
json_file = "experiments/simulation/s3/data2.ndjson"
df = pl.read_ndjson(json_file)

#df = df.filter(pl.col('single_client_data_fraction') == 0.5)
#print(len(df))
#df.write_ndjson(json_file)

grouping_keys = ['name', 'num_samples', 'single_client_data_fraction', 'alpha']
print(df.group_by(grouping_keys).agg(pl.len()))


#df1 = df.drop('fedci').rename({'pvalagg': 'metrics'}).with_columns(name=pl.lit('p_val_agg'))
#df2 = df.drop('pvalagg').rename({'fedci': 'metrics'}).with_columns(name=pl.lit('fedci'))
#pl.concat([df1,df2]).write_ndjson("experiments/simulation/s3/new_data.ndjson")

df = df.explode('metrics')
print(df.group_by('name').agg(pl.col('metrics').null_count()/pl.len()))

df = df.with_columns(
    pl.col('metrics').struct.unnest().name.prefix("metric_")
)

cols = cs.ends_with('_SHD') | cs.ends_with('_FOR') | cs.ends_with('_FDR')


df_agg = df.group_by(grouping_keys).agg(pl.len(), cols.mean()).sort(grouping_keys)
print(df_agg)
