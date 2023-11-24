import time as pytime
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import dolfin as df
import numpy as np
import pandas as pd
import pantarei as pr
import pint
import ufl
from dolfin import grad, inner
from loguru import logger


class ArtificialSASConcentration(df.Constant):
    def __init__(self, scale=0.25, t1=4.43e4, t2=8.5e4):
        super().__init__(0)
        self.s = scale
        self.t1 = t1
        self.t2 = t2

    def update(self, t: float) -> df.Function:
        newval = self.s * (-np.exp(-t / self.t1) + np.exp(-t / self.t2))
        self.assign(newval)

class ArtificialCSFConcentration:
    def __init__(self, sas, ventricles):
        self.inner = ventricles
        self.outer = sas
            
    def update(self, t: float) -> df.Function:
        self.inner.update(t)
        self.outer.update(t)

def repeated_assigner(u: df.Function, ui: df.Function):
    """Assigns to each component of a function u - residing in a vector function
    space W - a function ui residing in the component subspace V."""
    W = u.function_space()
    V = ui.function_space()
    n = W.num_sub_spaces()
    df.FunctionAssigner(W, [V] * n).assign(u, [ui] * n)
    return u


def solve_multidiffusion(
    u_data: pr.DataInterpolator,
    W: df.FunctionSpace,
    form: pr.TimedependentForm,
    coefficients: pr.CoefficientsDict,
    boundaries: list[pr.BoundaryData],
    time: pr.TimeKeeper,
    solver: pr.StationaryProblemSolver,
    storage: pr.FenicsStorage,
    total_storage: pr.FenicsStorage,
    total: Callable[[df.Function], df.Function],
    computer: Optional[pr.BaseComputer] = None,
):
    computer = pr.set_optional(computer, pr.NullComputer)

    if isinstance(u_data, (ArtificialSASConcentration, ArtificialCSFConcentration)):
        u0 = df.Function(W)
    else: 
        u0 = repeated_assigner(df.Function(W), u_data)

    storage.write_function(u0, "fluid_concentrations")
    uT = total(u0)
    computer.compute(time, uT)
    total_storage.write_function(uT, "total_concentration")

    dirichlet_bcs = pr.process_dirichlet(W, boundaries)

    F = form(W, coefficients, boundaries, u0, time.dt)
    a = df.lhs(F)
    l = df.rhs(F)
    A = df.assemble(a)

    u = df.Function(W, name="fluid_concentrations")
    tic = pytime.time()
    for ti in time:
        pr.print_progress(float(ti), time.endtime, rank=df.MPI.comm_world.rank)
        u_data.update(float(ti))
        b = df.assemble(l)
        solver.solve(u, A, b, dirichlet_bcs)

        uT = total(u)
        computer.compute(ti, uT)
        storage.write_checkpoint(u, "fluid_concentrations", float(ti))
        total_storage.write_checkpoint(uT, "total_concentration", float(ti))
        u0.assign(u)

    storage.close()
    total_storage.close()

    logger.info("Time loop finished.")
    toc = pytime.time()
    df.MPI.comm_world.barrier()
    logger.info(f"Elapsed time in loop: {toc - tic:.2f} seconds.")
    return computer


def solve_diffusion(
    u_data: pr.DataInterpolator,
    u0: df.Function,
    V: df.FunctionSpace,
    form: pr.TimedependentForm,
    coefficients: pr.CoefficientsDict,
    boundaries: list[pr.BoundaryData],
    time: pr.TimeKeeper,
    solver: pr.StationaryProblemSolver,
    storage: pr.FenicsStorage,
    computer: Optional[pr.BaseComputer] = None,
):
    computer = pr.set_optional(computer, pr.NullComputer)

    storage.write_function(u0, "total_concentration")
    computer.compute(time, u0)

    dirichlet_bcs = pr.process_dirichlet(V, boundaries)

    F = form(V, coefficients, boundaries, u0, time.dt)
    a = df.lhs(F)
    l = df.rhs(F)
    A = df.assemble(a)

    u = df.Function(V, name="total_concentration")
    tic = pytime.time()
    for ti in time:
        pr.print_progress(float(ti), time.endtime, rank=df.MPI.comm_world.rank)
        u_data.update(float(ti))
        b = df.assemble(l)
        solver.solve(u, A, b, dirichlet_bcs)
        computer.compute(ti, u)
        storage.write_checkpoint(u, "total_concentration", float(ti))
        u0.assign(u)

    storage.close()

    logger.info("Time loop finished.")
    toc = pytime.time()
    df.MPI.comm_world.barrier()
    logger.info(f"Elapsed time in loop: {toc - tic:.2f} seconds.")
    return computer

def multidiffusion_model(
    compartments: list[str],
    coefficients: pr.CoefficientsDict,
    inputfile: Path,
    outputfile: Path,
    totalfile: Optional[Path],
    method: str = "cg",
    preconditioner: str = "hypre_amg",
):
    # Needed to scale data from total- to fluid-concentrations
    phi = coefficients["phi"]
    phi_T = sum((phi[j] for j in compartments))

    # Read data for boundary conditions
    timevec, data = read_concentration_data(inputfile, "total_concentration")
    for di in data:
        di.vector().vec().array /= phi_T
    u_data = pr.DataInterpolator(data, timevec)

    # Setup timestepping
    dt = 3600
    T = timevec[-1]
    time = pr.TimeKeeper(dt=dt, endtime=T)

    # Extract functionspace and domain from data.
    V = u_data.function_space()
    domain = V.mesh()
    el = pr.read_signature(V.element().signature())

    # and build a vectorfunctionspace from it
    element = df.VectorElement(
        el.family(), el.cell(), el.degree(), dim=len(compartments)
    )
    W = df.FunctionSpace(domain, element)


    # If all Robin, use artificial boundary conditions.
    if all([0 < coefficients["robin"][j] < float('inf') for j in compartments]):
        logger.info("Using artificial boundaries")
        u_data = ArtificialCSFConcentration(
            sas = ArtificialSASConcentration(scale=0.52/phi_T),
            ventricles = ArtificialSASConcentration(scale=0.2/phi_T)
        )

    boundary_data = {
        idx: create_compartment_boundary(coefficients["robin"][j], u_data)
        for idx, j in enumerate(compartments)
    }

    print("Boundary (boundarydata): ", boundary_data)
    boundaries = pr.indexed_boundary_conditions(boundary_data)

    # Create computers
    dx = df.Measure("dx", domain, subdomain_data=domain.subdomains)
    computer = solute_quantifier(dx)

    if totalfile is None:
        total_storage = pr.NullStorage()
    else:
        total_storage = pr.FenicsStorage(totalfile, "w")

    time.reset()
    computer = solve_multidiffusion(
        u_data,
        W,
        form=partial(multicomp_diffusion_form, compartments=compartments),
        coefficients=coefficients,
        boundaries=boundaries,
        time=time,
        solver=pr.StationaryProblemSolver(method, preconditioner),
        storage=pr.FenicsStorage(outputfile, "w"),
        total_storage=total_storage,
        total=total_concentration(W, phi, compartments),
        computer=solute_quantifier(dx),
    )

    return computer


def create_compartment_boundary(coeff, value):
    if coeff == float("inf"):
        return [pr.DirichletBoundary(value, "everywhere")]
    elif coeff == 0.0:
        return []
    else:
        logger.info("Using artifical boundary data")
        return [pr.RobinBoundary(coeff, value.outer, (4, 5)), pr.RobinBoundary(coeff, value.inner, 8)]


def repeated_assigner(u: df.Function, ui: df.Function):
    """Assigns to each component of a function u - residing in a vector function
    space W - a function ui residing in the component subspace V."""
    W = u.function_space()
    V = ui.function_space()
    n = W.num_sub_spaces()
    df.FunctionAssigner(W, [V] * n).assign(u, [ui] * n)


def multicomp_diffusion_form(
    V: df.FunctionSpace,
    coefficients: pr.CoefficientsDict,
    boundaries: list[pr.BoundaryData],
    c0: df.Function,
    dt: float,
    compartments: list[str],
) -> df.Form:
    dx = df.Measure("dx", domain=V.mesh())
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    source = (
        coefficients["source"]
        if "source" in coefficients
        else {j: 0 for j in compartments}
    )
    cellform = (
        sum(
            (
                compartment_form(
                    idx_j,
                    u,
                    v,
                    c0,
                    coefficients,
                    compartments,
                    dt,
                    source,
                )
                for idx_j, _ in enumerate(compartments)
            )
        )
        * dx
    )
    return cellform + pr.process_boundary_forms(u, v, boundaries)


def compartment_form(
    idx_j: int,
    u: ufl.Argument,
    v: ufl.Argument,
    c0: df.Function,
    coefficients: pr.CoefficientsDict,
    compartments: list[str],
    dt: float,
    source: dict[str, float],
) -> df.Form:
    j = compartments[idx_j]
    phi, D, beta, r = tuple(coefficients[x] for x in ["phi", "D", "beta", "r"])
    sj = sum(
        [
            beta * (u[idx_i] - u[idx_j])
            for idx_i, i in enumerate(compartments)
            if idx_i != idx_j
        ]
    )
    return (
        phi[j]
        * (
            (u[idx_j] - c0[idx_j]) / dt * v[idx_j]
            + inner(D[j] * grad(u[idx_j]), grad(v[idx_j]))
        )
        + (r[j] * u[idx_j] - sj) * v[idx_j]
        - (source[j] * v[idx_j])
    )


def subspace_local_dofs(W: df.FunctionSpace, idx: int):
    mesh = W.mesh()
    dofs = W.sub(idx).dofmap().entity_closure_dofs(mesh, mesh.topology().dim())
    dofs = np.sort(np.unique(dofs))
    return dofs


def total_concentration(
    W: df.FunctionSpace, phi: dict[str, float], compartments: list[str]
):
    uT = df.Function(W.sub(0).collapse())
    N = len(compartments)
    dofs = [subspace_local_dofs(W, idx) for idx in range(N)]

    def call(u: df.Function):
        uT.vector().set_local(
            sum(
                (
                    phi[i] * u.vector().get_local(dofs[idx])
                    for idx, i in enumerate(compartments)
                )
            )
        )
        uT.vector().apply("insert")
        return uT

    return call


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input data file", type=str, required=True)
    parser.add_argument("--output", type=str, help="fluid_concs.hdf", required=True)
    parser.add_argument("--output_total", type=str, default=None)
    parser.add_argument("--De", type=float)
    parser.add_argument("--Dp", type=float)
    parser.add_argument("--phie", type=float)
    parser.add_argument("--phip", type=float)
    parser.add_argument("--tep", type=float)
    parser.add_argument("--tpb", type=float)
    parser.add_argument("--ke", type=float)
    parser.add_argument("--kp", type=float)
    parser.add_argument("--visual", action="store_true")
    args = parser.parse_args()

    data_file = Path(args.input)
    output = Path(args.output)

    param_units = {
        "phi": "",
        "D": "mm**2 / s",
        "r": "1 / s",
        "beta": "1 / s",
        "robin": "mm / s",
    }
    defaults = multidiffusion_parameters(param_units)

    # Override coefficients with CLI arguments.
    argmap = {
        "De": ("D", "ecs"),
        "Dp": ("D", "pvs"),
        "phie": ("phi", "ecs"),
        "phip": ("phi", "pvs"),
        "tep": ("beta"),
        "tpb": ("r", "pvs"),
        "ke": ("robin", "ecs"),
        "kp": ("robin", "pvs"),
    }

    # Set coefficients to default if not provided as argument.
    for arg, keys in filter(lambda x: getattr(args, x[0]) is not None, argmap.items()):
        nested_dict_set(defaults, keys, float(getattr(args, arg)))

    if df.MPI.comm_world.rank == 0:
        print()
        print("=== Coefficients: ===")
        print_quantities(defaults, offset=10)
        print()

    compartments = ["ecs", "pvs"]
    totalpath = Path(args.output_total) if args.output_total is not None else None
    computer = multidiffusion_model(
        compartments=compartments,
        coefficients=defaults,
        inputfile=Path(args.input),
        outputfile=Path(args.output),
        totalfile=args.output_total,
    )

    if df.MPI.comm_world.rank == 0:
        logger.info("Building dataframe from computer.")
        dframe = pd.DataFrame.from_dict(computer.values)
        logger.info("Storing dataframe to csv.")
        dframe.to_csv(Path(args.output).with_suffix(".csv"))

    df.MPI.comm_world.barrier()
    if args.visual:
        logger.info("Writing XDMF files for each compartment.")
        file = pr.FenicsStorage(output, "r")
        k = float_string_formatter(float(defaults["robin"]["ecs"]), 3)
        filenamer = lambda x: output.parent / (
            f"visual/multidiffusion-{x}.xdmf"# + f"_robin{k}" * (k != "inf") + ".xdmf"
        )
        file.to_xdmf(
            "fluid_concentrations",
            compartments,
            filenamer,
        )
        file.close()

        if args.output_total is not None:
            logger.info("Writing XDMF files for total concentration.")
            rel_path = "visual/multidiffusion" + f"_robin{k}" * (k != "inf") + ".xdmf"
            rel_path = "visual/multidiffusion.xdmf"
            file = pr.FenicsStorage(Path(args.output_total), "r")
            file.to_xdmf(
                "total_concentration",
                "total_concentration",
                lambda _: output.parent / rel_path,
            )
            file.close()
