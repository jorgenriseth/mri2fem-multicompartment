import numbers
from typing import Any, Optional

import pint

ureg = pint.get_application_registry()
dimensionless = pint.Unit("")
mm = ureg.mm
cm = ureg.cm
s = ureg.second
minute = ureg.minute


MULTIDIFFUSION_PARAMETERS = {
    "D_e": 1.3e-4 * mm**2 / s,
    "D_p": 3 * 1.3e-4 * mm**2 / s,
    "phi_e": 0.20 * dimensionless,
    "phi_p": 0.02 * dimensionless,
    "t_ep": 2.9e-2 * 1 / s,
    "t_pb": 0.21e-5 * 1 / s,
    "k_p": 3.7e-4 * mm / s,
    "k_e": 1.0e-5 * mm / s,
}

DEFAULT_UNITS = {
    "phi": "",
    "D": "mm**2 / s",
    "r": "1 / s",
    "beta": "1 / s",
    "robin": "mm / s",
}


def multidiffusion_parameters(param_units: Optional[dict[str, str]] = None):
    defaults = {**MULTIDIFFUSION_PARAMETERS}
    coefficients = {
        "phi": {"ecs": defaults["phi_e"], "pvs": defaults["phi_p"]},
        "D": {
            "ecs": defaults["D_e"],
            "pvs": defaults["D_p"],
        },
        "r": {
            "ecs": 0.0 * (1 / s),
            "pvs": defaults["t_pb"],
        },
        "beta": defaults["t_ep"],
        "robin": {
            "ecs": defaults["k_e"],
            "pvs": defaults["k_p"],
        },
    }
    if param_units is not None:
        return make_dimless(convert_to_units(coefficients, param_units))
    return coefficients


def twocomp_to_singlecomp_reduction(tc_params,  model: str):
    phi_e, phi_p = (tc_params[f"phi"][x] for x in ["ecs", "pvs"])
    D_e, D_p = (tc_params[f"D"][x] for x in ["ecs", "pvs"])
    t_pb = tc_params["r"]["pvs"]
    k_e, k_p = (tc_params[f"robin"][x] for x in ["ecs", "pvs"])
    if model == "fasttransfer":
        params = {
            "D": (phi_e * D_e + phi_p * D_p) / (phi_e + phi_p),
            "r": t_pb / (phi_e + phi_p),
            "robin": (k_e + k_p) / (phi_e + phi_p),
        }
    elif model == "singlecomp":
        params = {
            "D": D_e,
            "r": t_pb / phi_e,
            "robin": k_e / phi_e
        }
    else:
        raise ValueError(
            f"Invalid model '{model}' should be 'fasttransfer' or 'singlecomp'."
        )
    return params


def singlecomp_parameters(param_units: Optional[dict[str, str]] = None):
    defaults = multidiffusion_parameters(param_units)
    return twocomp_to_singlecomp_reduction(defaults, "singlecomp")


def fasttransfer_parameters(param_units: Optional[dict[str, str]] = None):
    defaults = multidiffusion_parameters(param_units)
    return twocomp_to_singlecomp_reduction(defaults, "fasttransfer")


def make_dimless(params):
    """Converts all quantities to a dimless number."""
    dimless = {}
    for key, val in params.items():
        if isinstance(val, dict):
            dimless[key] = make_dimless(val)
        else:
            dimless[key] = val.magnitude
    return dimless


def make_quantity_dimless(val: pint.Quantity) -> float:
    assert val.unitless, f"Parameter {val} not dimless"
    return (val.to_base_units()).magnitude


def convert_to_units(params, param_units):
    """Converts all quantities to the units specified by
    param_units."""
    # TODO: Make recursive to work with nested dictionaries
    converted = {}
    for key, val in params.items():
        if isinstance(val, dict):
            converted[key] = {}
            for j in val:
                converted[key][j] = val[j].to(param_units[key])
        elif isinstance(val, pint.Quantity):
            converted[key] = val.to(param_units[key])
        else:
            converted[key] = val
    return converted


def is_quantity(x: Any) -> bool:
    return isinstance(x, pint.Quantity) or isinstance(x, numbers.Complex)


def print_quantities(p, offset, depth=0):
    """Pretty printing of dictionaries filled with pint.Quantities (or numbers)"""
    format_size = offset - depth * 2
    for key, value in p.items():
        if isinstance(value, dict):
            print(f"{depth*'  '}{str(key)}")
            print_quantities(value, offset, depth=depth + 1)
        else:
            if is_quantity(value):
                print(f"{depth*'  '}{str(key):<{format_size+1}}: {value:.3e}")
            else:
                print(f"{depth*'  '}{str(key):<{format_size+1}}: {value}")



if __name__ == "__main__":
    import pprint

    print("=== Multidiffusion ===")
    print_quantities(multidiffusion_parameters(), 4)

    print()
    print("=== Diffusion - Single compartment===")
    print_quantities(singlecomp_parameters(), 4)

    print()
    print("=== Diffusion - Fast transfer ===")
    print_quantities(fasttransfer_parameters(), offset=4)
