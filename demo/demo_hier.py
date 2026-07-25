# demo_hier.py — exercise the hierarchical forms with Cars93 data
#
#   python demo/demo_hier.py

from pathlib import Path

import pandas as pd

from lessPy import Chart

here = Path(__file__).parent
d = pd.read_csv(Path.home() / "Cars93.csv")
d["Source"] = d["Source"].map({0: "nonUSA", 1: "USA"})

figs = {
    "hier_treemap":
        Chart("Type", by="Source", data=d, form="treemap"),
    "hier_sunburst":                       # pie + by = sunburst
        Chart("Type", by="Source", data=d, form="pie"),
    "hier_icicle":
        Chart("Type", y="MPGcity", stat="mean", data=d,
              form="icicle"),
}

for name, fig in figs.items():
    fig.write_html(here / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(here / f"{name}.png",
                        width=750, height=550, scale=2)
    except Exception as e:
        print(f"png skipped ({e})")
    print(f"wrote demo/{name}.html")
