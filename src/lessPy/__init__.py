# Python analog of NAMESPACE: names imported here are the public API.

from .bc_plotly import bc_plotly
from .bubble_plotly import bubble_plotly
from .Chart import Chart
from .dn_plotly import dn_plotly
from .dot_plotly import dot_plotly
from .freq_poly_plotly import freq_poly_plotly
from .hier_plotly import hier_plotly
from .hs_plotly import hs_plotly
from .Logit import Logit
from .pie_plotly import pie_plotly
from .plt_plotly import plt_plotly
from .radar_plotly import radar_plotly
from .Regression import Regression
from .utils import get_option, set_option
from .vbs_plotly import vbs_plotly
from .X import X
from .ANOVA import ANOVA
from .ttest import ttest
from .datasets import datasets, read_data
from .reshape import reshape_long, reshape_wide
from .pivot import pivot
from .corEFA import corEFA
from .corCFA import corCFA
from .corScree import corScree
from .corReorder import corReorder
from .corProp import corProp
from .Correlation import Correlation
from .corReflect import corReflect
from .corRead import corRead
from .Prop_test import Prop_test
from .corPrint import corPrint
from .Flows import Flows
from .date_infer import date_infer, format_date_labels
from .rename import rename
from .showColors import showColors
from .getColors import getColors
from .simCLT import simCLT
from .simMeans import simMeans
from .simFlips import simFlips
from .simCImean import simCImean
from .order_by import order_by
from .prob_norm import prob_norm
from .prob_znorm import prob_znorm
from .prob_tcut import prob_tcut
from .details import details
from .VariableLabels import VariableLabels
from .XY import XY

__version__ = "0.1.0"

__all__ = ["Chart", "X", "XY", "ANOVA", "ttest", "Regression", "Logit",
           "read_data", "datasets", "reshape_long", "reshape_wide", "pivot", "corEFA", "corCFA", "corScree", "corReorder", "corProp", "Correlation", "corReflect", "corRead", "Prop_test", "corPrint", "Flows", "date_infer", "format_date_labels", "rename", "showColors", "getColors", "simCLT", "simMeans", "simFlips", "simCImean", "order_by", "prob_norm", "prob_znorm", "prob_tcut", "details",
           "VariableLabels",
           "bc_plotly", "bubble_plotly", "dn_plotly", "dot_plotly",
           "freq_poly_plotly", "hier_plotly", "hs_plotly",
           "pie_plotly", "plt_plotly", "radar_plotly",
           "vbs_plotly",
           "get_option", "set_option"]
