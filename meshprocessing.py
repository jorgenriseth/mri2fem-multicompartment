#!/usr/bin/env python
import logging
from pathlib import Path
from typing import TypeAlias, Union

from dolfin import HDF5File, Mesh, MeshFunction

logger = logging.getLogger(__name__)


StrPath: TypeAlias = Union[str, Path]


def hdf2fenics(hdf5file, pack=False):
    """Function to read h5-file with annotated mesh, subdomains
    and boundaries into fenics mesh"""
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), str(hdf5file), "r") as hdf:
        hdf.read(mesh, "/domain/mesh", False)
        n = mesh.topology().dim()
        subdomains = MeshFunction("size_t", mesh, n)
        if hdf.has_dataset("/domain/subdomains"):
            hdf.read(subdomains, "/domain/subdomains")
        boundaries = MeshFunction("size_t", mesh, n - 1)
        if hdf.has_dataset("/domain/boundaries"):
            hdf.read(boundaries, "/domain/boundaries")

    if pack:
        return Domain(mesh, subdomains, boundaries)

    return mesh, subdomains, boundaries


class Domain(Mesh):
    def __init__(self, mesh: Mesh, subdomains: MeshFunction, boundaries: MeshFunction):
        super().__init__(mesh)
        self.subdomains = transfer_meshfunction(self, subdomains)
        self.boundaries = transfer_meshfunction(self, boundaries)


def transfer_meshfunction(newmesh: Mesh, meshfunc: MeshFunction) -> MeshFunction:
    newtags = MeshFunction("size_t", newmesh, dim=meshfunc.dim())  # type: ignore
    newtags.set_values(meshfunc)  # type: ignore
    return newtags
