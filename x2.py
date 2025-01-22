import polars as pl
import polars.selectors as cs

file = 'experiments/simulation/s1/data.ndjson'

df = pl.read_ndjson(file)

df_global = df.with_columns(pl.col('global').struct.unnest()).drop('global')
df_local = df.with_columns(pl.col('local').struct.unnest()).drop('local')

df_global = df_global.with_columns(pl.col('single').struct.unnest().name.suffix('_single'))
df_global = df_global.with_columns(pl.col('coop').struct.unnest().name.suffix('_coop'))

df_local = df_local.with_columns(pl.col('single').struct.unnest().name.suffix('_single'))
df_local = df_local.with_columns(pl.col('coop').struct.unnest().name.suffix('_coop'))

#df = df.explode(cs.ends_with('_single_global') - cs.starts_with('num_pags'))
#df = df.explode(cs.ends_with('_coop_global') - cs.starts_with('num_pags'))

print(df_local.select(cs.starts_with('shd'), cs.starts_with('fdr'), cs.starts_with('for')).describe())
