import argparse
import datetime
import json
from pathlib import Path
from typing import Callable, List, Optional

import dolfin as df
import nibabel
import numpy
import numpy as np
from nibabel.affines import apply_affine
from panta_rhei import FenicsStorage


def read_image(
    filename: Path,
    functionspace: df.FunctionSpace,
    data_filter: Optional[
        Callable[[np.ndarray, np.ndarray, int, int, int], np.ndarray]
    ] = None,
):
    mri_volume = nibabel.load(filename)
    voxeldata: np.ndarray = mri_volume.get_fdata()  # type: ignore
    ras2vox_tkr_inv = numpy.linalg.inv(mri_volume.header.get_vox2ras_tkr())

    c_data = df.Function(functionspace, name="concentration")
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


def image_timestamp(p: Path) -> datetime.datetime:
    return datetime.datetime.strptime(p.stem, "%Y%m%d_%H%M%S")


def injection_timestamp(injection_time_file: Path, subjectid: str) -> datetime.datetime:
    with open(injection_time_file, "r") as f:
        time_string = json.load(f)[subjectid]
    return datetime.datetime.strptime(time_string.strip(), "%Y%m%d_%H%M%S")


def fenicsstorage2xdmf(
    filepath, funcname: str, subnames: str | List[str], outputdir: Path
) -> None:
    file = FenicsStorage(filepath, "r")
    file.to_xdmf(funcname, subnames, outputdir)
    file.close()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import numpy as np
    from panta_rhei.fenicsstorage import FenicsStorage
    from panta_rhei.meshprocessing import hdf2fenics

    parser = argparse.ArgumentParser()
    parser.add_argument("--meshfile", type=Path, required=True)
    parser.add_argument("--concentrations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--femfamily", type=str, default="CG")
    parser.add_argument("--femdegree", type=int, default=1)
    args = parser.parse_args()

    meshpath = Path(args.meshfile)
    domain = hdf2fenics(meshpath, pack=True)
    V = df.FunctionSpace(domain, args.femfamily, args.femdegree)

    output = Path(args.output)
    concentration_data = args.concentrations
    t = np.loadtxt(args.timestamps)
    outfile = FenicsStorage(str(output), "w")
    outfile.write_domain(domain)
    for ti, cfile in zip(t, concentration_data):
        c_data_fenics = read_image(filename=cfile, functionspace=V, data_filter=None)
        outfile.write_checkpoint(c_data_fenics, name="total_concentration", t=ti)
    outfile.close()

    fenicsstorage2xdmf(
        outfile.filepath,
        "total_concentration",
        "total_concentration",
        lambda _: outfile.filepath.parent / "visual/data.xdmf",  # type: ignore
    )
