import polars as pl
import panel as pn
import hvplot.polars
import hvplot

df = pl.read_parquet("./error-data-01.parquet")

plot  = df.cast(pl.Int32).hvplot.box("X", ["Y", "Z"])

plot_display = pn.panel(plot).show(port=40010)
plot_display.stop()

plot  = df.cast(pl.Int32).hvplot.box("Y", ["X", "Z"])

plot_display = pn.panel(plot).show(port=40010)
plot_display.stop()
