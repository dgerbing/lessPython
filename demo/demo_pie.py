# demo_pie.py — exercise Chart(form="pie") with the Cars93 data
#
#   python demo/demo_pie.py

from pathlib import Path

import pandas as pd

from lessPy import Chart, pie_plotly

here = Path(__file__).parent
d = pd.read_csv(Path.home() / "Cars93.csv")
d["Source"] = d["Source"].map({0: "nonUSA", 1: "USA"})

figs = {
    "pie_counts":                          # auto-title, % labels
        Chart("Type", data=d, form="pie"),
    # Chart(by=) on a pie now renders a sunburst (as in lessR);
    # the pie GRID is R's facet= treatment — until Chart() gains
    # facet=, build the grid by calling the renderer directly
    "pie_by":
        pie_plotly(pd.crosstab(d["Source"], d["Type"]),
                   digits_d=0, main="Cars93: Type by Source"),
    "pie_mean":                            # aggregation, input labels
        Chart("Type", y="MPGcity", stat="mean", data=d,
              form="pie"),
}

for name, fig in figs.items():
    fig.write_html(here / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(here / f"{name}.png",
                        width=700, height=550, scale=2)
    except Exception as e:
        print(f"png skipped ({e})")
    print(f"wrote demo/{name}.html")
