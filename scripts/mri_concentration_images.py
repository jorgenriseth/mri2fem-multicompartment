# %%
from pathlib import Path
import re

import matplotlib.cm as cm
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import nibabel
import numpy as np
from twocomp.utils import is_T1_mgz
from twocomp.utils import orient_and_slice_image


def create_model_filter(model):
    def filter_(p):
        return re.match(f"^{model}_", p.name) is not None

    return filter_



def crop_image(im, ymin=0, ymax=None, xmin=0, xmax=None):
    return im[ymin:ymax, xmin:xmax]


def concentration_mri_reader(iterator, orientation, crop_idx):  # , cutoff):
    ims = []
    for idx, p in enumerate(iterator):
        im_data = nibabel.load(p).get_fdata()
        im = orient_and_slice_image(
            im_data,
            orientation,
            slice_idx,
        )
        im = crop_image(im, *crop_idx)
        ims.append(im)
    return ims


def filter_concentrations(path):
    return re.match(r"concentration_\d.mgz", path.name) is not None


# %%
image_orientation = "transversal"
slice_idx = 100
crop_idx = (45, 245, 30, 230)

reference_image_path = Path("data/concentration_0.mgz")
reference_image_data = nibabel.load(reference_image_path).get_fdata()
uncropped = orient_and_slice_image(reference_image_data, image_orientation, slice_idx)
reference_image = crop_image(uncropped, *crop_idx)

timestamp_seconds = np.loadtxt("data/timestamps.txt")
hours = np.rint(timestamp_seconds / 3600).astype(int)

mri_concentration_filter = create_model_filter("concentration")
mri_concentration_paths = filter(
    mri_concentration_filter, sorted(Path("data").iterdir())
)
concentration_images = concentration_mri_reader(
    mri_concentration_paths, image_orientation, crop_idx
)

twocomp_filter = create_model_filter("multidiffusion_total")
twocomp_mris = list(filter(twocomp_filter, sorted(Path("results/mri").iterdir())))
twocomp_images = concentration_mri_reader(twocomp_mris, image_orientation, crop_idx)

color_range = (-0.05, 0.15)
cbar_width = 50
sizes = [x.shape for x in concentration_images]
total_width = sum([min(x[0], x[1]) for x in sizes]) + cbar_width
total_height = max([x[1] for x in sizes])

fig_width = 12
scale_factor = fig_width / total_width
fig_height = scale_factor * total_height

offsets = [0, *np.cumsum([max(x[0], x[1]) for x in sizes]) / total_width]
image_widths = np.diff(offsets)

fig = plt.figure(figsize=(fig_width, fig_height))
for idx, im in enumerate(concentration_images):
    ax = fig.add_axes([offsets[idx], 0.0, image_widths[idx], 1.0])
    ax.imshow(reference_image, cmap="gray")
    ax.imshow(im, cmap="magma", vmin=color_range[0], vmax=color_range[1])
    ax.set_xticks([])
    ax.set_yticks([])

ax = fig.add_axes([offsets[-1], 0.0, 1 - offsets[-1], 1.0])
cmap = cm.magma
norm = mplcolors.Normalize(vmin=color_range[0], vmax=color_range[1])

# Create colorbar
c = fig.colorbar(
    cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax,
    orientation="vertical",
    label="Concentration [mM]",
)
c.set_label(label="", size=40)
ax.tick_params(axis="y", labelsize=16)
ax.title.set_size(24)
fig.savefig("figures/mri_concentration_row.pdf", bbox_inches="tight")
fig.savefig("figures/mri_concentration_row.png", bbox_inches="tight")
fig.show()

cbar_width = 100
sizes = [x.shape for x in twocomp_images]
total_width = sum([min(x[0], x[1]) for x in sizes]) + cbar_width
total_height = max([x[1] for x in sizes])

fig_width = 12
scale_factor = fig_width / total_width
fig_height = scale_factor * total_height

offsets = [0, *np.cumsum([max(x[0], x[1]) for x in sizes]) / total_width]
image_widths = np.diff(offsets)

fig = plt.figure(figsize=(fig_width, fig_height))
for idx, im in enumerate(twocomp_images):
    ax = fig.add_axes([offsets[idx], 0.0, image_widths[idx], 1.0])
    ax.imshow(reference_image, cmap="gray")
    ax.imshow(im, cmap="magma", vmin=color_range[0], vmax=color_range[1])
    ax.set_xticks([])
    ax.set_yticks([])

ax = fig.add_axes([offsets[-1], 0.0, (1 - offsets[-1]) * 0.5, 1.0])
cmap = cm.magma
norm = mplcolors.Normalize(vmin=color_range[0], vmax=color_range[1])

# Create colorbar
c = fig.colorbar(
    cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax,
    orientation="vertical",
    label="Concentration (mM)",
)
c.set_label(label="Concentration (mM)", size=19)
ax.tick_params(axis="y", labelsize=16)
fig.savefig("figures/twocomp_diffusion_row.pdf", bbox_inches="tight")
fig.savefig("figures/twocomp_diffusion_row.png", bbox_inches="tight")

plt.show()
