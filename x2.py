import polars as pl
import polars.selectors as cs

file = 'experiments/simulation/s1/data_test_alot.ndjson'

df = pl.read_ndjson(file)

df = df.with_columns(pl.col('single').struct.unnest().name.suffix('_single'))
df = df.with_columns(pl.col('coop').struct.unnest().name.suffix('_coop'))

df = df.drop('single', 'coop')

print(df.select('num_pags_single', 'num_pags_coop').describe())

df = df.explode(cs.ends_with('_single') - cs.starts_with('num_pags'))
df = df.explode(cs.ends_with('_coop') - cs.starts_with('num_pags'))

print(df.select(cs.starts_with('shd'), cs.starts_with('fdr'), cs.starts_with('for')).describe())
