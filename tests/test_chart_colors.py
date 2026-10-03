import pytest

from lessPy import Chart, read_data
from lessPy.plotly_utils import to_hex


# Bar fills golden against lessR 4.5.7 (with its Oct 2026 theme fix) (Employee data, x = Dept): the
# first 7 hex digits of each trace's colors, one list per trace. A
# trace whose bars share one color is given that color once.
GOLDEN = {
    'D1': ({'fill': 'reds'},
            [['#E8AFBA', '#CB8492', '#AD5A6C', '#8F2C48', '#730024']]),
    'D2': ({'by': 'Plan', 'fill': 'reds'},
            [['#CD949F'], ['#A85467'], ['#810032']]),
    'D3': ({'fill': 'viridis'},
            [['#4B0055', '#00588B', '#009B95', '#53CC67', '#FDE333']]),
    'C1': ({'theme': 'gray'},
            [['#595959', '#595959', '#595959', '#595959', '#595959']]),
    'C2': ({'theme': 'darkred'},
            [['#8B0000', '#8B0000', '#8B0000', '#8B0000', '#8B0000']]),
    'C3': ({'by': 'Gender', 'theme': 'gray'},
            [['#4D4D4D'], ['#999999']]),
    'C4': ({'by': 'Plan', 'theme': 'darkgreen'},
            [['#004D00'], ['#437C25'], ['#8AAD7F']]),
    'E1': ({'y': 'Salary', 'stat': 'mean', 'fill_scaled': True},
            [['#0064A5', '#004B90', '#005A9C', '#00599B', '#004E92']]),
    'E2': ({'y': 'Salary', 'stat': 'mean', 'fill_scaled': True, 'theme': 'darkred'},
            [['#A82C2C', '#880000', '#9C1E1E', '#9A1B1B', '#8C0102']]),
    'E3': ({'y': 'Salary', 'stat': 'mean', 'fill_scaled': True, 'fill': 'darkgreen'},
            [['#0D6910', '#004E00', '#005F00', '#005D00', '#005200']]),
    'E4': ({'y': 'Salary', 'stat': 'mean', 'fill_scaled': True, 'theme': 'gray'},
            [['#5A5A5A', '#404040', '#505050', '#4F4F4F', '#434343']]),
    'C5': ({'by': 'Gender', 'theme': 'darkred'},
            [['#890C3A'], ['#BF8791']]),
    'C6': ({'by': 'Plan', 'theme': 'white'},
            [['#3E3E3E'], ['#6E6E6E'], ['#A3A3A3']]),
    'D4': ({'fill': 'plasma'},
            [['#001889', '#91008D', '#D24E71', '#EDA200', '#DAFF47']]),
    'D5': ({'by': 'Gender', 'fill': 'viridis'},
            [['#4B0055'], ['#FDE333']]),
}


def _hex7(c):
    c = str(c)
    if c.startswith("rgba"):
        r, g, b, _ = (float(v) for v in c[5:-1].split(","))
        return "#%02X%02X%02X" % (round(r), round(g), round(b))
    return to_hex(c)[:7].upper()


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_bar_fill_matches_lessR(case):
    kw, want = GOLDEN[case]
    fig = Chart("Dept", data=read_data("Employee"), quiet=True, **kw)
    got = []
    for t in fig.data:
        cols = (t.marker.color if isinstance(t.marker.color,
                                             (list, tuple))
                else [t.marker.color])
        hs = [_hex7(c) for c in cols]
        got.append(hs if (len(set(hs)) > 1 or len(fig.data) == 1)
                   else hs[:1])
    assert got == want
