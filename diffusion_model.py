import logging
from pathlib import Path

import dolfin as df
from dolfin import inner, grad

from boundary import DirichletBoundary, process_dirichlet
from fenicsstorage import FenicsStorage
from interpolator import vectordata_interpolator
from timekeeper import TimeKeeper
from multidiffusion_model import read_concentration_data
from utils import print_progress


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
df.set_log_level(df.LogLevel.WARNING)


def diffusion(coefficients, inputfile, outputfile=None):
    if outputfile is None:
        outputfile = Path(inputfile)
    logger.info(f"Reading data from {inputfile}")
    timevec, data = read_concentration_data(inputfile)
    interpolator = vectordata_interpolator(data, timevec)
    u0 = data[0].copy(deepcopy=True)
    u_interp = u0.copy(deepcopy=True)

    # Get functionspace and domain.
    V = u0.function_space()
    domain = V.mesh()

    # Set up timestepping
    dt = 3600
    T = timevec[-1]
    time = TimeKeeper(dt=dt, endtime=T)

    # Define boundary-conditions.
    boundaries = [DirichletBoundary(u_interp, "everywhere")]
    bcs = process_dirichlet(V, domain, boundaries)

    # Define variational problem
    D = coefficients["diffusion_coefficient"]
    dx = df.Measure("dx", V.mesh())
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    F = ((u - u0) * v + dt * (inner(D * grad(u), grad(v)))) * dx  # type: ignore
    a = df.lhs(F)
    l = df.rhs(F)
    A = df.assemble(a)

    #  Store initial condition.
    u = df.Function(V)
    u.assign(u0)
    storage = FenicsStorage(outputfile, "a")
    storage.write_function(u, "diffusion", overwrite=True)

    time.reset()
    for ti in time:
        print_progress(float(ti), time.endtime, rank=df.MPI.comm_world.rank)
        u_interp.vector()[:] = interpolator(float(ti))
        b = df.assemble(l)
        for bc in bcs:
            bc.apply(A, b)
        df.solve(A, u.vector(), b, "gmres", "hypre_amg")
        storage.write_checkpoint(u, "diffusion", float(ti))
        u0.assign(u)
    storage.close()

    return storage.filepath


if __name__ == "__main__":
    datapath = Path("data/")

    coefficients = {"diffusion_coefficient": 3.4e-4}  # mm^2/s
    results_path = diffusion(coefficients, datapath / "data.hdf", outputfile=datapath /"test.hdf")

    file = FenicsStorage(results_path, "r")
    file.to_xdmf("diffusion", "diffusion")
    file.close()
