import numbers
import re
from pathlib import Path
from typing import Optional


def float_string_formatter(x: float, digits: int):
    """Converts a numbers to a path-friendly floating points format without punctuation.
    ex: (1.3344, 3) -> 133e-3"""
    if float(x) == float("inf"):
        return "inf"
    return f"{x*10**(-digits):{f'.{digits}e'}}".replace(".", "")


def to_scientific(num: numbers.Complex, decimals: int) -> str:
    if float(num) == float("inf"):
        return "\infty"
    x = f"{float(num):{f'.{decimals}e'}}"
    m = re.search("(\d\.{0,1}\d*)e([\+|\-]\d{2})", x)

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
