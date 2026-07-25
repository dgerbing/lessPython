# demo_bubble.py — exercise Chart(form="bubble") with Cars93 data
#
#   python demo/demo_bubble.py

from pathlib import Path

import pandas as pd

from lessPy import Chart

here = Path(__file__).parent
d = pd.read_csv(Path.home() / "Cars93.csv")
d["Source"] = d["Source"].map({0: "nonUSA", 1: "USA"})
d["Airbags"] = d["Airbags"].map(
    {0: "none", 1: "driver", 2: "both"})

figs = {
    "bubble_counts":
        Chart("Type", data=d, form="bubble"),
    "bubble_matrix":
        Chart("Type", by="Airbags", data=d, form="bubble"),
    "bubble_mean":
        Chart("Type", y="MPGcity", stat="mean", data=d,
              form="bubble"),
}

for name, fig in figs.items():
    fig.write_html(here / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(here / f"{name}.png",
                        width=800, height=500, scale=2)
    except Exception as e:
        print(f"png skipped ({e})")
    print(f"wrote demo/{name}.html")
