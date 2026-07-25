# demo_xy.py — exercise XY() with the Cars93 data
#
#   python demo/demo_xy.py

from pathlib import Path

import numpy as np
import pandas as pd

from lessPy import XY

here = Path(__file__).parent
d = pd.read_csv(Path.home() / "Cars93.csv")
d["Source"] = d["Source"].map({0: "nonUSA", 1: "USA"})

# monthly series for the time-series display
rng = np.random.default_rng(3)
ts = pd.DataFrame({
    "Month": pd.date_range("2023-01-31", periods=30, freq="ME"),
    "Sales": (120 + np.arange(30) * 3
              + rng.normal(0, 9, 30)).round(1),
})

figs = {
    "xy_scatter":
        XY("Weight", "MPGhiway", data=d,
           main="XY(Weight, MPGhiway)"),
    "xy_fit":
        XY("Weight", "MPGhiway", data=d, fit="lm",
           main='XY(Weight, MPGhiway, fit="lm")'),
    "xy_ellipse":
        XY("Weight", "MPGhiway", data=d, ellipse=0.95,
           main="XY(Weight, MPGhiway, ellipse=0.95)"),
    "xy_by":
        XY("Weight", "MPGhiway", by="Source", data=d, fit="lm",
           main='XY(Weight, MPGhiway, by=Source, fit="lm")'),
    "xy_exp":
        XY("Weight", "MPGhiway", data=d, fit="exp",
           main='XY(Weight, MPGhiway, fit="exp")'),
    "xy_ts":
        XY("Month", "Sales", data=ts,
           main="XY(Month, Sales) — time series"),
}

for name, fig in figs.items():
    fig.write_html(here / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(here / f"{name}.png",
                        width=800, height=500, scale=2)
    except Exception as e:
        print(f"png skipped ({e})")
    print(f"wrote demo/{name}.html")
