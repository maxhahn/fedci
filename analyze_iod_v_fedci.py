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
json_file = "experiments/simulation/s3/data.ndjson"

# Load and prepare the data
df = load_and_prepare_data(json_file)
print(df.select(cs.ends_with('SHD'), cs.ends_with('FDR'), cs.ends_with('FOR')))

# Analyze fedci vs. pvalagg
expanded_df, mean_diffs = analyze_fedci_vs_pvalagg(df)

# Display the results
print("Expanded DataFrame:")
print(expanded_df)

print("\nMean Differences Between fedci and pvalagg:")
print(mean_diffs)

print(expanded_df.select(cs.starts_with('diff_')).describe())
