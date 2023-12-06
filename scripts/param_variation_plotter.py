# %%
import re
import itertools
from pathlib import Path
from typing import Callable
import numbers

import numpy as np
import matplotlib.pyplot as plt
import pint
import pandas as pd

from twocomp.utils import (
    parameter_dict_string_formatter,
    to_scientific,
    is_T1_mgz,
    parameter_regex_search_string,
)

# %%

ureg = pint.get_application_registry()
M_unit = (ureg("mmolars") * ureg("mm^3")).to("mmol")
M_scale = M_unit.magnitude
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# TODO: Import this from somewhere
base_params = {
    "De": 1.3e-4,
    "Dp": 3 * 1.3e-4,
    "phie": 0.20,
    "phip": 0.02,
    "tep": 2.9e-2,
    "tpb": 0.21e-5,
    "ke": 1.0e-5,
    "kp": 3.7e-4,
}


region_lims = {"whole-brain": 0.10, "white-matter": 0.06, "gray-matter": 0.06}
ccycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def is_csv(x: Path) -> bool:
    return x.suffix == ".csv"


# %%
def plot_variation(
    param_name: str,
    inputdir: Path,
    region: str,
    title: str,
    labler: Callable[[numbers.Complex], str],
    ax=None,
    plotkwargs=None,
):
    filter_string = parameter_regex_search_string([param_name], base_params, decimals=2)
    paths = sorted(filter(is_csv, inputdir.iterdir()))
    filelist = []
    paramlist = []
    for p in paths:
        m = re.match(filter_string, p.name)
        if m is not None:
            paramlist.append(m.groups()[0])
            filelist.append(p)

    if ax is None:
        _, ax = plt.subplots(1, 1)
    if plotkwargs is None:
        plotkwargs = {}

    tmax = 0
    for param, filename in sorted(zip(paramlist, filelist), key=lambda x: x[0]):
        tc_dframe = pd.read_csv(filename)
        tc_time = tc_dframe["time"] / 3600
        tc_content = tc_dframe[region] * M_scale
        ax.plot(tc_time, tc_content, lw=2, **plotkwargs)
        tmax = max(tmax, tc_time.iloc[-1])

    ax.set_title(title)
    ax.set_ylim(0, region_lims[region])
    ax.set_xlim(0, tmax)
    ax.spines[["right", "top"]].set_visible(False)
    legend_handles = [
        plt.Line2D(xdata=[0], ydata=[0], c=ccycle[idx], label=labler(param))
        for idx, param in enumerate(paramlist)
    ]
    legend_anchor = (1, 1.0)
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=legend_anchor,
        frameon=False,
    )
    return ax


plot_variation(
    param_name="Dp",
    inputdir=Path("data/single_param/twocomp"),
    region="whole-brain",
    title="$D_p$ (mm$^2$/s)",
    labler=lambda Dp: f"$D_p = {to_scientific(Dp, 2)}$",
)
plt.show()

# %%
region = "whole-brain"
plt.rcParams.update({"font.size": 20, "legend.fontsize": 20})
fig, axes = plt.subplots(3, 2, figsize=(16, 8), sharex=True, sharey=True)
Dp_scales = [3, 10, 100]
De_ref = 1.3e-4

plot_variation(
    param_name="Dp",
    inputdir=Path("data/single_param/twocomp"),
    region=region,
    title="$D_p^\\text{eff}$ (mm$^2$/s)",
    #     labler = lambda Dp: f"$D_p = {to_scientific(Dp, 2)}$ mm$^2$/s",
    labler=lambda Dp: f"${int(np.rint(float(Dp)/De_ref))}\\times D_e^\\text{{eff}}$",
    ax=axes[0, 0],
)
plot_variation(
    param_name="phip",
    inputdir=Path("data/single_param/twocomp"),
    region=region,
    title="$n_p$ (-)",
    labler=lambda x: f"${float(x):.2f}$",
    ax=axes[0, 1],
)

plot_variation(
    param_name="tep",
    inputdir=Path("data/single_param/twocomp"),
    region=region,
    title="$t_{{ep}}$ (s$^{-1}$)",
    labler=lambda x: f"${to_scientific(x, 2)}$",  # s$^{{-1}}$",
    ax=axes[1, 0],
)

plot_variation(
    param_name="tpb",
    inputdir=Path("data/single_param/twocomp"),
    region=region,
    title="$t_{{pb}}$ (s$^{-1}$)",
    labler=lambda x: f"${to_scientific(x, 2)}$",  # " s$^{{-1}}$",
    ax=axes[1, 1],
)

plot_variation(
    param_name="ke",
    inputdir=Path("data/single_param/twocomp"),
    region=region,
    title="$k_e$ (mm/s)",
    labler=lambda x: f"${to_scientific(x, 2)}$",  # mm/s",
    ax=axes[2, 0],
)

plot_variation(
    param_name="kp",
    inputdir=Path("data/single_param/twocomp"),
    region=region,
    title="$k_p$ (mm/s)",
    labler=lambda x: f"${to_scientific(x, 2)}$",  # mm/s",
    ax=axes[2, 1],
)

for i in range(axes.shape[0]):
    axes[i, 1].set_ylabel("")
for j in range(axes.shape[1]):
    axes[-1, j].set_xlabel("Time (h)")
axes[1, 0].set_ylabel("Total tracer content (mmol)", fontsize=30)

plt.tight_layout()
plt.savefig("figures/parameter-variations.pdf", bbox_inches="tight")
plt.show()
# %%
