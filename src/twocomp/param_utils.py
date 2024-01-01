import itertools
import re
from functools import partial
from typing import Mapping, Optional, Sequence

model_param_map = {
    "twocomp": ["De", "Dp", "phie", "phip", "tep", "tpb", "ke", "kp"],
    "fasttransfer": ["De", "Dp", "phie", "phip", "tpb", "ke", "kp"],
    "singlecomp": ["De", "phie", "ke", "tpb"],
}


def float_string_formatter(x: float, decimals: int):
    """Converts a numbers to a path-friendly floating points format without punctuation.
    ex: (1.3344, 3) -> 1334e-3"""
    if float(x) == float("inf"):
        return "inf"
    return f"{x*10**(-decimals):{f'.{decimals}e'}}".replace(".", "")


def parameter_string_to_named_value(param_string: str) -> tuple[str, float]:
    """Convert a single-parameter string to a tuple (param_name, param_value).
    ex: a1334e-3 -> ("a", 1.3344)"""
    ff = r"\d+e[\+|-]\d+|inf"
    match = re.match(rf"(\w+?)({ff})", param_string)
    if match is None:
        raise ValueError(f"Could not parse parameter string {param_string}")
    return str(match.groups()[0]), float(match.groups()[1])


def parameter_str_to_dict(param_string: str) -> dict[str, float]:
    """Converts parameter-string to dictionary of parameters
    ex: a1334e-3_b1000e-3 -> {"a": 1.3344, "b": 1.0}"""
    matches = [parameter_string_to_named_value(x) for x in param_string.split("_")]
    return {key: val for (key, val) in matches}


def parameter_dict_string_formatter(d: Mapping[str, float], decimals: int) -> str:
    """Convert parameter-dictionary to string_format
    ex: {"a": 1.3344, "b": 1.0} -> a1334e-3_b1000e-3"""
    fsformat = partial(float_string_formatter, decimals=decimals)
    key_val_pairs = [f"{key}{fsformat(val)}" for key, val in d.items()]
    return "_".join(key_val_pairs)


def to_scientific(num: float | str, decimals: int) -> str:
    """Convert floating-point number to latex-friendly floating-point format
    for e.g. plot-labelsh plotting: (0.0032, 2) -> 3.20\\times10^{-3}"""
    floatnum = float(num)
    if floatnum == float("inf"):
        return r"\infty"
    x = f"{floatnum:{f'.{decimals}e'}}"
    m = re.search(r"(\d\.{0,1}\d*)e([\+|\-]\d{2})", x)
    if m is None:
        raise ValueError(f"Regex expression not found in {x}")
    return f"{m.group(1)}\\times10^{{{int(m.group(2))}}}"


def create_parameter_variations(
    variations: Mapping[str, Sequence[float]],
    baseline: Mapping[str, float],
    model_params: Optional[Sequence[str]] = None,
) -> list[dict[str, float]]:
    """Create list of parameter sets, given baseline parameter dict, and a
    dictionary of all values to be tested for each of the parameters. If
    model_params are given, then any parameter not in the list are set to the
    baseline values.
    ex: ({"a": [1, 2], "c": [1, 2]}, {"a": 0, "b": 0, "c: 0}, ["a", "b"])
        -> [ {"a": 1, "b": 0, "c": 0}, {"a": 2, "b": 0, "c": 0} ]
    """
    if model_params is None:
        model_params = list(baseline.keys())

    products = itertools.product(
        *[variations[key] for key in variations]  # if key in model_params]
    )
    parameter_settings = []
    for product in products:
        new_setting = {**baseline}
        for key, val in zip(variations, product):
            if key in model_params:
                new_setting[key] = val
        parameter_settings.append(new_setting)
    return parameter_settings


def parameter_variation_strings(
    variation: Mapping[str, Sequence[float]],
    baseline: Mapping[str, float],
    decimals: int,
    model_params: Optional[Sequence[str]] = None,
) -> list[str]:
    """Takes a list of parameter-variations and the base-parameter setting,
    and creates a list of 2-decimal string-represented parameter sets.
    ex: ({"a": [1, 2]}, {"a": 0, "b": 0}, 2) -> ["a100e-2_b000e-2", "a200e-2_b000e-2"]
    """
    parameter_dicts = create_parameter_variations(variation, baseline, model_params)
    return list(
        set(
            [
                parameter_dict_string_formatter(param_dict, decimals)
                for param_dict in parameter_dicts
            ]
        )
    )


def create_variation(
    variation: Mapping[str, Sequence[float]], baseline: Mapping[str, float]
) -> Sequence[tuple[str, str]]:
    return [
        (modelname, paramset)
        for modelname in model_param_map
        for paramset in parameter_variation_strings(
            variation, baseline, 2, model_param_map[modelname]
        )
    ]


def model_fname_list(variation, baseline):
    return [
        f"{modelname}/{fname}"
        for modelname, fname in create_variation(variation, baseline)
    ]


def parameter_set_reduction(
    paramdict: Mapping[str, float],
    model_params: Sequence["str"],
    baseline_parameters: Mapping[str, float],
) -> dict[str, float]:
    out = {**baseline_parameters}
    for key in model_params:
        out[key] = paramdict[key]
    return out


def parameter_regex_search_string(
    free: Sequence[str], baseline: Mapping[str, float], decimals
) -> str:
    ff = r"\d+e[\+|-]\d+|inf"
    out_dict = {}
    for key, val in baseline.items():
        if key in free:
            out_dict[key] = f"({ff})"
        else:
            out_dict[key] = re.sub(r"\+", r"\+", float_string_formatter(val, decimals))
    return "_".join([f"{key}{val}" for key, val in out_dict.items()])
