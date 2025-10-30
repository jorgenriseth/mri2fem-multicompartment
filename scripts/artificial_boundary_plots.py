from pathlib import Path

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panta_rhei as pr
from matplotlib.lines import Line2D

# %%
plt.rcParams.update({"font.size": 16, "legend.fontsize": 16})
inputdir = Path("results/")
dframe = pd.read_csv("results/quantities.csv", index_col=0)
domain, subdomain, boundaries = pr.hdf2fenics("data/data.hdf", pack=False)

ds = df.Measure("ds", domain=domain, subdomain_data=boundaries)
pial_area = df.assemble(1.0 * ds(4))
ventricle_surf_area = df.assemble(1.0 * ds(8))
inferior_surf_area = df.assemble(1.0 * ds(5))
all_area = df.assemble(1.0 * ds)

t = dframe["time"].to_numpy()
y_average = (dframe["pial-surf"] + dframe["ventricle-surf"]) / (
    pial_area + ventricle_surf_area
)
ccycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# %%
def g(t, s, a, b):
    """General form of boundary-concentration."""
    return s * (-np.exp(-t / a) + np.exp(-t / b))


x = np.linspace(0, 1, 201) * t[-1]
t1, t2 = 4.3e4, 8.51e4
s1 = 0.52
s2 = 0.25
phi = 0.22

fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=False)
ticks = np.arange(0, np.rint(t[-1] / 3600 + 1))[::20].astype(int)

axes[0].plot(
    dframe["time"], dframe["pial-surf"] / pial_area, "--x", label=r"$u_{d,sas}$"
)
axes[0].plot(
    dframe["time"],
    dframe["ventricle-surf"] / ventricle_surf_area,
    "--x",
    label=r"$u_{d,vent}$",
)
axes[0].plot(x, g(x, s1 / phi, t1, t2), c=ccycle[2])
axes[0].plot(x, g(x, s2 / phi, t1, t2), c=ccycle[3])
axes[0].plot(x, g(x, s1, t1, t2), c=ccycle[0])
axes[0].plot(x, g(x, s2, t1, t2), c=ccycle[1])

axes[1].plot(dframe["time"], dframe["pial-surf"] / pial_area, "--x")
axes[1].plot(dframe["time"], dframe["ventricle-surf"] / ventricle_surf_area, "--x")
axes[1].plot(x, g(x, s1, t1, t2), c=ccycle[0])
axes[1].plot(x, g(x, s2, t1, t2), c=ccycle[1])
axes[1].plot(x, g(x, s1 / phi, t1, t2), c=ccycle[2])
axes[1].plot(x, g(x, s2 / phi, t1, t2), c=ccycle[3])

axes[0].set_ylim(0, None)
axes[1].set_ylim(0, 0.15)
axes[0].set_ylabel("Concentration (mM)")

for ax in axes:
    ax.set_xticks(ticks * 3600, ticks)
    ax.set_xlabel("Time (h)")
    ax.set_xlim(0, 72 * 3600)


legend_handles = [
    Line2D(
        xdata=[0], ydata=[0], c=ccycle[0], ls="--", marker="x", label=r"$u_{d,sas}$"
    ),
    Line2D(
        xdata=[0], ydata=[0], c=ccycle[1], ls="--", marker="x", label=r"$u_{d,vent}$"
    ),
    Line2D(xdata=[0], ydata=[0], c=ccycle[0], label=r"$\hat c_{sas}$"),
    Line2D(xdata=[0], ydata=[0], c=ccycle[1], label=r"$\hat c_{vent}$"),
    Line2D(xdata=[0], ydata=[0], c=ccycle[2], label=r"$\phi\hat c_{sas}$"),
    Line2D(xdata=[0], ydata=[0], c=ccycle[3], label=r"$\phi\hat c_{vent}$"),
]
axes[1].legend(
    handles=legend_handles, loc="upper left", bbox_to_anchor=(1, 1), frameon=False
)
plt.tight_layout()
fig.savefig("figures/sas-concentrations.pdf")  # , bbox_inches="tight")
plt.show()
