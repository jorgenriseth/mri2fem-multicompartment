import itertools
import re
from pathlib import Path

import dolfin as df
import numpy as np
import pantarei as pr


def float_string_formatter(x: float, decimals: int):
    """Converts a numbers to a path-friendly floating points format without punctuation.
    ex: (1.3344, 3) -> 1334e-3"""
    if float(x) == float("inf"):
        return "inf"
    return f"{x*10**(-decimals):{f'.{decimals}e'}}".replace(".", "")


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
