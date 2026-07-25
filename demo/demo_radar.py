# demo_radar.py — exercise Chart(form="radar") with the Cars93 data
#
#   python demo/demo_radar.py

from pathlib import Path

import pandas as pd

from lessPy import Chart

here = Path(__file__).parent
d = pd.read_csv(Path.home() / "Cars93.csv")
d["Source"] = d["Source"].map({0: "nonUSA", 1: "USA"})

figs = {
    "radar_counts":
        Chart("Type", data=d, form="radar"),
    "radar_by":
        # Cars93 has no nonUSA Large cars, an empty cell that radar
        # correctly rejects (as in lessR) — exclude that category
        Chart("Type", by="Source", data=d, form="radar",
              filter="Type != 'Large'"),
    "radar_mean":
        Chart("Type", y="MPGcity", stat="mean", data=d,
              form="radar"),
}

for name, fig in figs.items():
    fig.write_html(here / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(here / f"{name}.png",
                        width=700, height=550, scale=2)
    except Exception as e:
        print(f"png skipped ({e})")
    print(f"wrote demo/{name}.html")
