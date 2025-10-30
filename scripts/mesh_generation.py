import subprocess
from pathlib import Path

import SVMTK as svmtk
from panta_rhei import mesh2xdmf, xdmf2hdf
from loguru import logger


def create_patient_mesh(meshpath: Path, resolution: int) -> Path:
    surfaces = ("lh.pial", "rh.pial", "lh.white", "rh.white", "ventricles")
    surfaces = [meshpath / surf for surf in surfaces]
    stls = [meshpath / f"{surf.name}.stl" for surf in surfaces]
    for surface, stl in zip(surfaces, stls):
        if "ventricles" in surface.name:
            continue
        subprocess.run(f"mris_convert {surface} {stl}", shell=True)
    return create_brain_mesh(
        stls,
        meshpath / "brain.mesh",
        resolution,
        remove_ventricles=True,
    )


def create_ventricle_surface(patientdir: Path) -> Path:
    input = patientdir / "mri/wmparc.mgz"
    output = patientdir / "MODELING/ventricles.stl"
    if not output.exists():
        subprocess.run(
            f"bash scripts/extract-ventricles.sh {input} {output}", shell=True
        )
    return output


def create_brain_mesh(
    stls: list[Path], output: Path, resolution: int, remove_ventricles: bool = True
) -> Path:
    """Taken from original mri2fem-book:
    https://github.com/kent-and/mri2fem/blob/master/mri2fem/mri2fem/chp4/fullbrain-five-domain.py
    """
    logger.info(f"Creating brain mesh from surfaces {stls}")
    surfaces = [svmtk.Surface(str(stl)) for stl in stls]
    # FIXME: Remove numbered references as this is very change sensitive
    # Merge lh rh surface, and drop the latter.
    surfaces[2].union(surfaces[3])
    surfaces.pop(3)

    # Define identifying tags for the different regions
    tags = {"pial": 1, "white": 2, "ventricle": 3}

    # Label the different regions
    smap = svmtk.SubdomainMap()
    smap.add("1000", tags["pial"])
    smap.add("0100", tags["pial"])
    smap.add("1010", tags["white"])
    smap.add("0110", tags["white"])
    smap.add("1110", tags["white"])
    smap.add("1011", tags["ventricle"])
    smap.add("0111", tags["ventricle"])
    smap.add("1111", tags["ventricle"])

    # Generate mesh at given resolution
    domain = svmtk.Domain(surfaces, smap)
    domain.create_mesh(resolution)

    # Remove ventricles perhaps
    if remove_ventricles:
        domain.remove_subdomain(tags["ventricle"])

    # Save mesh
    domain.save(str(output.with_suffix(".mesh")))
    xdmfdir = output.parent / "mesh_xdmfs"
    xdmfdir.mkdir(exist_ok=True)
    mesh2xdmf(output.with_suffix(".mesh"), xdmfdir, dim=3)
    return xdmf2hdf(xdmfdir, output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--surfaces", nargs="+", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    args = parser.parse_args()
    surfaces = ["lh.pial", "rh.pial", "lh.white", "rh.white", "ventricles"]
    stls = [Path(stl) for stl in args.surfaces]
    meshfile = create_brain_mesh(
        stls=stls,
        output=Path(args.output),
        resolution=args.resolution,
        remove_ventricles=True,
    )
    logger.info(f"Generated mesh in file {meshfile}")
