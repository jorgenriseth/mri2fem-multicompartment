import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import dolfin as df
import nibabel
import numpy
from nibabel.affines import apply_affine
from multidiffusion.fenicsstorage import FenicsStorage


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_image(filename, functionspace, data_filter=None):
    mri_volume = nibabel.load(filename)
    voxeldata = mri_volume.get_fdata()

    c_data = df.Function(functionspace, name="concentration")
    ras2vox_tkr_inv = numpy.linalg.inv(mri_volume.header.get_vox2ras_tkr())

    xyz = functionspace.tabulate_dof_coordinates()
    ijk = apply_affine(ras2vox_tkr_inv, xyz).T
    i, j, k = numpy.rint(ijk).astype("int")

    if data_filter is not None:
        voxeldata = data_filter(voxeldata, ijk, i, j, k)
        c_data.vector()[:] = voxeldata[i, j, k]
    else:
        if numpy.where(numpy.isnan(voxeldata[i, j, k]), 1, 0).sum() > 0:
            print(
                "No filter used, setting",
                numpy.where(numpy.isnan(voxeldata[i, j, k]), 1, 0).sum(),
                "/",
                i.size,
                " nan voxels to 0",
            )
            voxeldata[i, j, k] = numpy.where(
                numpy.isnan(voxeldata[i, j, k]), 0, voxeldata[i, j, k]
            )
        if numpy.where(voxeldata[i, j, k] < 0, 1, 0).sum() > 0:
            print(
                "No filter used, setting",
                numpy.where(voxeldata[i, j, k] < 0, 1, 0).sum(),
                "/",
                i.size,
                " voxels in mesh have value < 0",
            )

        c_data.vector()[:] = voxeldata[i, j, k]

    return c_data


def image_timestamp(p: Path) -> datetime:
    return datetime.strptime(p.stem, "%Y%m%d_%H%M%S")


def injection_timestamp(injection_time_file: Path) -> datetime:
    with open(injection_time_file, "r") as f:
        time_string = f.read()
    return datetime.strptime(time_string, "%H.%M.%S").time()


def fenicsstorage2xdmf(filepath, funcname: str, subnames: str | List[str]) -> None:
    file = FenicsStorage(filepath, "r")
    file.to_xdmf(funcname, subnames)
    file.close()
