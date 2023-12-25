import itertools
import re
import dolfin as df
import pantarei as pr
import numpy as np
from pathlib import Path


def float_string_formatter(x: float, decimals: int):
    """Converts a numbers to a path-friendly floating points format without punctuation.
    ex: (1.3344, 3) -> 1334e-3"""
    if float(x) == float("inf"):
        return "inf"
    return f"{x*10**(-decimals):{f'.{decimals}e'}}".replace(".", "")


def parameter_dict_string_formatter(d: dict[str, complex], decimals: int) -> str:
    fsformat = lambda x: float_string_formatter(x, decimals)
    key_val_pairs = [f"{key}{fsformat(val)}" for key, val in d.items()]
    return "_".join(key_val_pairs)


def to_scientific(num: complex, decimals: int) -> str:
    floatnum = float(num)  # type: ignore
    if floatnum == float("inf"):
        return r"\infty"
    x = f"{floatnum:{f'.{decimals}e'}}"
    m = re.search(r"(\d\.{0,1}\d*)e([\+|\-]\d{2})", x)
    if m is None:
        raise ValueError(f"Regex expression not found in {x}")
    return f"{m.group(1)}\\times10^{{{int(m.group(2))}}}"


def nested_dict_set(
    d: dict[str, dict | float], keys: tuple[str], value: float
) -> dict[str, dict | float]:
    if isinstance(keys, str):
        d[keys] = value
        return d

    depth = len(keys)
    d_ = d
    for i, key in enumerate(keys):
        if i == depth - 1:
            d_[key] = value
        else:
            d_ = d[key]
    return d


def create_parameter_variations(
    variations, baseline, model_params=None
) -> list[dict[str, complex]]:
    if model_params is None:
        model_params = baseline.keys()
    keys = variations.keys()
    products = itertools.product(*[variations[key] for key in keys])

    parameter_settings = []
    for product in products:
        new_setting = {**baseline}
        for key, val in zip(keys, product):
            new_setting[key] = val
        parameter_settings.append(new_setting)
    return parameter_settings


def read_concentration_data(filepath, funcname) -> tuple[np.ndarray, list[df.Function]]:
    store = pr.FenicsStorage(filepath, "r")
    tvec = store.read_timevector(funcname)
    c = store.read_function(funcname, idx=0)
    C = [df.Function(c.function_space()) for _ in range(tvec.size)]
    for idx in range(len(C)):
        store.read_checkpoint(C[idx], funcname, idx)
    return tvec, C


def solute_quantifier(dx):
    return pr.BaseComputer(
        {
            "whole-brain": lambda u: df.assemble(u * dx),
            "gray-matter": lambda u: df.assemble(u * dx(1)),
            "white-matter": lambda u: df.assemble(u * dx(2)),
        }
    )


def is_T1_mgz(p: Path) -> bool:
    return re.match("[0-9]{8}_[0-9]{6}.mgz$", p.name) is not None


# %%
def parameter_set_reduction(
    paramdict: dict[str, complex],
    model_params: list["str"],
    baseline_parameters: dict[str, complex],
) -> dict[str, complex]:
    out = {**baseline_parameters}
    for key in model_params:
        out[key] = paramdict[key]
    return out


# %%
def parameter_regex_search_string(
    free: list[str], baseline: dict[str, float], decimals
) -> str:
    ff = r"\d+e[\+|-]\d+|inf"
    out_dict = {}
    for key, val in baseline.items():
        if key in free:
            out_dict[key] = f"({ff})"
        else:
            out_dict[key] = re.sub(r"\+", r"\+", float_string_formatter(val, decimals))
    return "_".join([f"{key}{val}" for key, val in out_dict.items()])


def parameter_str_to_dict(pstring):
    ff = r"\d+e[\+|-]\d+|inf"
    return {
        key: value
        for (key, value) in map(
            lambda x: re.match(f"(\w+?)({ff})", x).groups(), pstring.split("_")
        )
    }


def orient_and_slice_image(
    data: np.ndarray, orientation: str, slice_idx: int, cutoff: float = None
):
    if orientation == "saggital":
        plane = np.s_[slice_idx, :, :]
        im = data[plane]
    elif orientation == "transversal":
        plane = np.s_[:, slice_idx, :]
        im = np.rot90(data[plane])
    elif orientation == "coronal":
        plane = np.s_[:, :, slice_idx]
        im = data[plane].T
    else:
        raise ValueError(f"wrong orientation, got {orientation}")
    if cutoff is not None:
        im[abs(im) <= cutoff] = np.nan
    return im
