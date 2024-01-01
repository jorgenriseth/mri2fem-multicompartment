import pint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

ureg = pint.get_application_registry()
M_unit = (ureg("mmolars") * ureg("mm^3")).to("mmol")
M_scale = M_unit.magnitude
ccycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plt.rcParams.update({"font.size": 15, "legend.fontsize": 16})
modelingdir = Path("results/mri_boundary")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), sharey=True)
for idx, region in enumerate(["whole-brain", "gray-matter", "white-matter"]):
    ax = axes[idx]

    sc1_dframe = pd.read_csv(modelingdir / "diffusion.csv", index_col=0)
    sc1_time = sc1_dframe["time"] / 3600
    sc1_content = sc1_dframe[region] * M_scale
    ax.plot(sc1_time, sc1_content, lw=3)

    tc_dframe = pd.read_csv(modelingdir / "multidiffusion.csv", index_col=0)
    tc_time = tc_dframe["time"] / 3600
    tc_content = tc_dframe[region] * M_scale
    ax.plot(tc_time, tc_content, lw=2)

    sc2_dframe = pd.read_csv(modelingdir / "diffusion_ecs_only.csv", index_col=0)
    sc2_time = sc2_dframe["time"] / 3600
    sc2_content = sc2_dframe[region] * M_scale
    ax.plot(sc2_time, sc2_content, lw=2)

    data_dframe = pd.read_csv(modelingdir / "../quantities.csv")
    data_time = data_dframe["time"] / 3600
    data_content = data_dframe[region] * M_scale
    ax.plot(data_time, data_content, "kx", markersize=8)

    ax.set_ylim(0, 0.08)
    ax.set_xlim(0, None)
    ax.set_xticks(np.arange(0, 71, 20))
    ax.set_title(region.replace("-", " ").title())
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel("Time (h)")

    print()
    print("Region:", region)
    print("Peaks:")
    peaks = np.array(
        [tc_content.max(), sc1_content.max(), sc2_content.max(), data_content.max()]
    )
    print(peaks[:, np.newaxis] / peaks)

    print("Tail")
    tail = np.array(
        [
            tc_content.iloc[-1],
            sc1_content.iloc[-1],
            sc2_content.iloc[-1],
            data_content.iloc[-1],
        ]
    )
    print(tail[:, np.newaxis] / tail)


axes[0].set_ylabel("Total tracer\ncontent (mmol)", fontsize=20)

legend_handles = [
    plt.Line2D([0], [0], color=ccycle[0], label=f"(TC)"),
    plt.Line2D([0], [0], color=ccycle[1], label=f"(SC1)"),
    plt.Line2D([0], [0], color=ccycle[2], label=f"(SC2)"),
    plt.Line2D([0], [0], c="k", marker="x", label="Data"),
]
fig.legend(
    loc="outside right upper",
    handles=legend_handles,
    bbox_to_anchor=(1.05, 0.95),
    frameon=False,
)

plt.tight_layout()
plt.savefig("figures/regionwise-models-data.pdf", bbox_inches="tight")
plt.show()
