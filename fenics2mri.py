"""
Adjusted from Bastian Zapf's code in 
https://github.com/bzapf/mritracer/blob/main/computetracer/fenics2mri.py
"""

import argparse
import itertools
import logging
from pathlib import Path

import nibabel
import numpy as np
from nibabel.affines import apply_affine
from tqdm import tqdm

from multidiffusion.fenicsstorage import FenicsStorage
from multidiffusion.interpolator import interpolate_from_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def function_to_image(
    function, template_image, extrapolation_value, mask
) -> tuple[nibabel.Nifti1Image, np.ndarray]:
    shape = template_image.get_fdata().shape
    output_data = np.zeros(shape) + extrapolation_value
    vox2ras = template_image.header.get_vox2ras_tkr()
    V = function.function_space()

    ## Code to get a bounding box for the mesh, used to not iterate over all the voxels in the image
    imap = V.dofmap().index_map()
    num_dofs_local = imap.local_range()[1] - imap.local_range()[0]
    xyz = V.tabulate_dof_coordinates()
    xyz = xyz.reshape((num_dofs_local, -1))
    image_coords = apply_affine(np.linalg.inv(vox2ras), xyz)

    lower_bounds = np.maximum(0, np.floor(image_coords.min(axis=0)).astype(int))
    upper_bounds = np.minimum(shape, np.ceil(image_coords.max(axis=0)).astype(int))

    all_relevant_indices = itertools.product(
        *(range(start, stop + 1) for start, stop in zip(lower_bounds, upper_bounds))
    )
    num_voxels_in_mask = np.product(1 + upper_bounds - lower_bounds)
    fraction_of_image = num_voxels_in_mask / np.product(shape)
    print(
        f"Computed mesh bounding box, evaluating {fraction_of_image:.0%} of all image voxels"
    )
    print(f"There are {num_voxels_in_mask} voxels in the bounding box")

    # Populate image
    def eval_fenics(f, coords, extrapolation_value):
        try:
            return f(*coords)
        except RuntimeError:
            return extrapolation_value

    eps = 1e-12

    progress = tqdm(total=num_voxels_in_mask)

    for xyz_vox in all_relevant_indices:
        xyz_ras = apply_affine(
            vox2ras, xyz_vox
        )  # transform_coords(coords, vox2ras, inverse=True)
        output_data[xyz_vox] = eval_fenics(function, xyz_ras, extrapolation_value)
        progress.update(1)

    output_data = np.where(output_data < eps, eps, output_data)

    # Save output
    output_nii = nibabel.Nifti1Image(
        output_data, template_image.affine, template_image.header
    )

    return output_nii, output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5_name", type=str, help="Name of function inside the HDF5 file"
    )
    parser.add_argument("--extrapolation_value", type=float, default=float("nan"))
    args = parser.parse_args()

    datapath = Path("data")
    nii_img = nibabel.load(datapath / "T1.mgz")
    storage = FenicsStorage(datapath / "data.hdf", "r")
    timevec = storage.read_timevector("cdata")
    for idx, ti in enumerate(timevec):
        ci = interpolate_from_file(storage.filepath, args.hdf5_name, ti)
        hours = int(round(ti / 3600))
        logging.info(f"Processing date at timepoint {hours} hours")
        output_volume, output_arry = function_to_image(
            function=ci,
            template_image=nii_img,
            extrapolation_value=args.extrapolation_value,
            mask=args.mask,
        )

        output_path = Path(f"results/{args.hdf5_name}_{hours:02d}.nii.gz")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        nibabel.save(output_volume, output_path)
