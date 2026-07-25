# demo_chart.py — exercise Chart() with the Cars93 data
#
#   python demo/demo_chart.py
#
# Writes chart_*.html (and .png if kaleido is installed) into demo/.

from pathlib import Path

import pandas as pd

from lessPy import Chart

here = Path(__file__).parent
d = pd.read_csv(Path.home() / "Cars93.csv")
d["Source"] = d["Source"].map({0: "nonUSA", 1: "USA"})

figs = {
    "chart_counts":
        Chart("Type", data=d, sort="-",
              main="Chart(Type)"),
    "chart_by":
        Chart("Type", by="Source", data=d, sort="-",
              main="Chart(Type, by=Source)"),
    "chart_mean":
        Chart("Type", y="MPGcity", stat="mean", data=d, sort="-",
              main='Chart(Type, y=MPGcity, stat="mean")'),
    "chart_dev":
        Chart("Type", y="MPGcity", stat="deviation", data=d,
              sort="-", horiz=True,
              main='Chart(Type, y=MPGcity, stat="deviation")'),
}

for name, fig in figs.items():
    fig.write_html(here / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(here / f"{name}.png",
                        width=800, height=500, scale=2)
    except Exception as e:
        print(f"png skipped ({e})")
    print(f"wrote demo/{name}.html")
