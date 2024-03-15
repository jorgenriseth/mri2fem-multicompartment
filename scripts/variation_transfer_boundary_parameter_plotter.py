import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
from matplotlib.lines import Line2D

from twocomp.param_utils import (
    parameter_dict_string_formatter,
    parameter_regex_search_string,
    to_scientific,
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
region = "whole-brain"
inputdir = Path("results/compartmentalization")

paths = sorted(filter(is_csv, (inputdir / "twocomp").iterdir()))
filter_string = parameter_regex_search_string(["tep", "ke"], base_params, decimals=2)
filelist = []
paramlist = []
variations = {"tep": set(), "ke": set()}
for p in paths:
    m = re.match(filter_string, p.name)
    if m is not None:
        paramlist.append(m.groups())
        filelist.append(p)
        variations["tep"].add(m.groups()[0])
        variations["ke"].add(m.groups()[1])
for key, val in variations.items():
    variations[key] = sorted(set(val))


n = len(variations["ke"])
fig, axes = plt.subplots(1, n, figsize=(12, 3.5))
for i, ke in enumerate(variations["ke"]):
    d = {**base_params}
    d["ke"] = float(ke)
    ax = axes[i]
    sc2_dframe = pd.read_csv(
        inputdir / f"singlecomp/{parameter_dict_string_formatter(d, 2)}.csv",
        index_col=0,
    )
    sc2_time = sc2_dframe["time"] / 3600
    sc2_content = sc2_dframe[region] * M_scale
    ax.plot(sc2_time, sc2_content, ls="-.", c="k")
    sc1_dframe = pd.read_csv(
        inputdir / f"fasttransfer/{parameter_dict_string_formatter(d, 2)}.csv",
        index_col=0,
    )
    sc1_time = sc1_dframe["time"] / 3600
    sc1_content = sc1_dframe[region] * M_scale
    ax.plot(sc1_time, sc1_content, c="k", ls=":", lw=3)
    time_max = max(sc1_time.iloc[-1], sc2_time.iloc[-1])
    for idx, tep in enumerate(variations["tep"]):
        d["tep"] = float(tep)
        tc_dframe = pd.read_csv(
            inputdir / f"twocomp/{parameter_dict_string_formatter(d, 2)}.csv",
            index_col=0,
        )
        tc_time = tc_dframe["time"] / 3600
        tc_content = tc_dframe[region] * M_scale
        ax.plot(tc_time, tc_content, c=ccycle[idx])
        time_max = max(time_max, tc_time.iloc[-1])

    ax.set_title(f"$k_e = {to_scientific(ke, 2)} mm/s$", fontsize=20)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Total tracer \n content (mmol)", fontsize=18)
    ax.set_ylim(0, region_lims[region])
    ax.set_xlim(0, time_max)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xticks(np.arange(0, 71, 20))


legend_handles = [
    Line2D([0], [0], c=ccycle[idx], label=f"$t_{{ep}}={to_scientific(param, 2)}$")
    for idx, param in enumerate(variations["tep"])
] + [
    Line2D([0], [0], c="k", ls=":", label="SC1"),
    Line2D([0], [0], c="k", ls="-.", label="SC2"),
]

for i in range(n):
    if i > 0:
        axes[i].set_yticks(axes[i].get_yticks(), [])
        axes[i].set_ylabel(None)

    if i == n - 1:
        fig.legend(
            loc="outside upper right",
            handles=legend_handles,
            bbox_to_anchor=(1.25, 1.0),
            fontsize=18,
            frameon=False,
        )

plt.tight_layout()
plt.savefig("figures/varying_tep_ke.pdf", bbox_inches="tight")
plt.show()
# %%
