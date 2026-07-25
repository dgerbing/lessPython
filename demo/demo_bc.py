# demo_bc.py — exercise bc_plotly() with the Cars93 data
#
# Tabulation happens here, in the caller — the same division of
# labor as lessR, where Chart()/X() aggregate and bc_plotly renders.
#
#   python demo/demo_bc.py
#
# Writes bc_1d.html, bc_2d.html (and .png if kaleido is installed)
# into the demo/ directory.

from pathlib import Path

import pandas as pd

from lessPy import bc_plotly

here = Path(__file__).parent
d = pd.read_csv(Path.home() / "Cars93.csv")
d["Source"] = d["Source"].map({0: "nonUSA", 1: "USA"})

# ---- 1-D: counts of one categorical variable -----------------------
x1 = d["Type"].value_counts()          # Series: index=Type, values=n
fig1 = bc_plotly(x1, x_name="Type", y_name="Count",
                 digits_d=0, main="Cars93: Count of Type")

# ---- 2-D: crosstab, by variable in rows -----------------------------
x2 = pd.crosstab(d["Source"], d["Type"])
x2 = x2[x1.index]                      # columns in descending order
fig2 = bc_plotly(x2, digits_d=0,
                 main="Cars93: Type by Source")

for name, fig in [("bc_1d", fig1), ("bc_2d", fig2)]:
    fig.write_html(here / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(here / f"{name}.png",
                        width=800, height=500, scale=2)
    except Exception as e:
        print(f"png skipped ({e})")
    print(f"wrote demo/{name}.html")
