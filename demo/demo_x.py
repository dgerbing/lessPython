# demo_x.py — exercise X() with the Cars93 data
#
#   python demo/demo_x.py

from pathlib import Path

import pandas as pd

from lessPy import X

here = Path(__file__).parent
d = pd.read_csv(Path.home() / "Cars93.csv")
d["Source"] = d["Source"].map({0: "nonUSA", 1: "USA"})

figs = {
    "x_hist":
        X("MPGcity", data=d, main="X(MPGcity)"),
    "x_hist_by":
        X("MPGcity", by="Source", data=d,
          main="X(MPGcity, by=Source)"),
    "x_density":
        X("MPGcity", data=d, form="density",
          main='X(MPGcity, form="density")'),
    "x_density_by":
        X("MPGcity", by="Source", data=d, form="density",
          main='X(MPGcity, by=Source, form="density")'),
    "x_vbs":
        X("MPGcity", data=d, form="vbs",
          main='X(MPGcity, form="vbs")'),
    "x_vbs_full":
        X("MPGcity", data=d, form="vbs", vbs_mean=True,
          fences=True,
          main="X(MPGcity, vbs_mean=TRUE, fences=TRUE)"),
    "x_box":
        X("MPGcity", data=d, form="box",
          main='X(MPGcity, form="box")'),
    "x_vbs_by":
        X("MPGcity", by="Source", data=d, form="vbs",
          main='X(MPGcity, by=Source, form="vbs")'),
    "x_vbs_facet":
        X("MPGcity", facet="Type", data=d, form="vbs",
          main='X(MPGcity, facet=Type, form="vbs")'),
    "x_hist_facet":
        X("MPGcity", facet="Source", data=d,
          main="X(MPGcity, facet=Source)"),
    "x_density_facet":
        X("MPGcity", facet="Source", data=d, form="density",
          main='X(MPGcity, facet=Source, form="density")'),
}

for name, fig in figs.items():
    fig.write_html(here / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(here / f"{name}.png",
                        width=800, height=500, scale=2)
    except Exception as e:
        print(f"png skipped ({e})")
    print(f"wrote demo/{name}.html")
