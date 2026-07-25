# demo_dot.py — exercise Chart(form="dot") with the Cars93 data
#
#   python demo/demo_dot.py

from pathlib import Path

import pandas as pd

from lessPy import Chart

here = Path(__file__).parent
d = pd.read_csv(Path.home() / "Cars93.csv")

figs = {
    "dot_counts":
        Chart("Type", data=d, form="dot", sort="-"),
    "dot_mean":
        Chart("Type", y="MPGcity", stat="mean", data=d,
              form="dot", sort="-", horiz=True),
    "dot_paired":
        Chart("Model", y=["MPGcity", "MPGhiway"], data=d.head(12),
              form="dot", sort="-"),
}

for name, fig in figs.items():
    fig.write_html(here / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(here / f"{name}.png",
                        width=800, height=500, scale=2)
    except Exception as e:
        print(f"png skipped ({e})")
    print(f"wrote demo/{name}.html")
