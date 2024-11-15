import polars as pl
import hvplot.polars
import hvplot
import panel as pn
import glob

# Load data
path =  './experiments/expanded_ordinals/*ndjson'
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

df = df.with_columns(experiment_type=pl.col('name').str.slice(0,3))
print(df['experiment_type'].value_counts())
df = df.explode('predicted_p_values', 'true_p_values')

# Plot scatter of p values
plot1 = df.hvplot.scatter(
    x='predicted_p_values',
    y='true_p_values',
    alpha=0.7,
    ylim=(-0.1,1.1),
    xlim=(-0.1,1.1),
    height=400,
    #width=400,
    by='name', width=600,
    groupby=['experiment_type', 'num_clients', 'num_samples']
    )

#hvplot.show(plot, port=8080)
#hvplot.save(plot, 'images/p_value_scatter.html')
plot_display = pn.panel(plot1).show(port=40010)
plot_display.stop()

# Plot correlation of p values
_df = df
_df = _df.group_by('name', 'experiment_type', 'num_clients', 'num_samples') \
    .agg(pl.corr('predicted_p_values', 'true_p_values')) \
    .rename({'predicted_p_values': 'p_value_correlation'})

plot2 = _df.sort('num_samples').hvplot.line(x='num_samples',
                                  y='p_value_correlation',
                                  alpha=0.6,
                                  by='name',
                                  groupby=['experiment_type','num_clients'],
                                  ylim=(0.5,1.001)
                                  )
#hvplot.show(plot, port=8080)
#hvplot.save(plot, 'images/p_value_corr.html')
plot_display = pn.panel(plot2).show(port=40010)
plot_display.stop()

# Plot accuracy
alpha = 0.05

_df = df
_df = _df.with_columns(
    tp=(pl.col('predicted_p_values') < alpha) & (pl.col('true_p_values') < alpha),
    tn=(pl.col('predicted_p_values') > alpha) & (pl.col('true_p_values') > alpha),
    fp=(pl.col('predicted_p_values') < alpha) & (pl.col('true_p_values') > alpha),
    fn=(pl.col('predicted_p_values') > alpha) & (pl.col('true_p_values') < alpha),
)

_df = _df.group_by('name', 'experiment_type', 'num_clients', 'num_samples').agg((pl.col('tp')+pl.col('tn')).mean().alias('accuracy'))

plot3 = _df.sort('num_samples').hvplot.line(x='num_samples',
                                  y='accuracy',
                                  alpha=0.6,
                                  by='name',
                                  groupby=['experiment_type','num_clients'],
                                  ylim=(0.5,1.001)
                                  )
#hvplot.show(plot, port=8080)
#hvplot.save(plot, 'images/p_value_acc.html')
plot_display = pn.panel(plot3).show(port=40010)
plot_display.stop()

#total_plot = (plot1+plot2+plot3).cols(1)
#plot_display = pn.panel(total_plot).show(port=8080)
#plot_display.stop()
