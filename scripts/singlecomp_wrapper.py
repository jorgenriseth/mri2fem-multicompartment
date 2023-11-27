""" Wrapper-script to more easily run the diffusion-model script
with parameters derived from the two-compartment model."""
import subprocess
from pathlib import Path

from loguru import logger

import twocomp.diffusion as diffusion_module
from twocomp.parameters import (
    multidiffusion_parameters,
    twocomp_to_singlecomp_reduction,
)
from twocomp.utils import nested_dict_set

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--De", type=float)
    parser.add_argument("--Dp", type=float)
    parser.add_argument("--phie", type=float)
    parser.add_argument("--phip", type=float)
    parser.add_argument("--tep", type=float)
    parser.add_argument("--tpb", type=float)
    parser.add_argument("--ke", type=float)
    parser.add_argument("--kp", type=float)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--model", type=str, default="fasttransfer") 
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    # Potentially override defaults parameters from command line
    param_units = {
        "phi": "",
        "D": "mm**2 / s",
        "r": "1 / s",
        "beta": "1 / s",
        "robin": "mm / s",
    }
    tc_params = multidiffusion_parameters(param_units)
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
    for arg, keys in filter(lambda x: getattr(args, x[0]) is not None, argmap.items()):
        nested_dict_set(tc_params, keys, float(getattr(args, arg)))

    from twocomp.parameters import print_quantities
    coefficients = twocomp_to_singlecomp_reduction(tc_params, args.model)
    print()
    print("=== Coefficients: ===")
    print_quantities(coefficients, offset=10)
    print()

    script = Path(diffusion_module.__file__).relative_to(Path(".").resolve())
    cmd = (
        f"mpirun -n {args.threads}"
        f" python '{script}'"
        f" --input '{args.input}'"
        f" --output '{args.output}'"
        f" --D {coefficients['D']}"
        f" --r {coefficients['r']}"
        f" --k {coefficients['robin']}"
    )
    logger.info(f"Executing '{cmd}'")
    subprocess.run(cmd, shell=True)
