import re

from twocomp.param_utils import (
    create_parameter_variations,
    parameter_dict_string_formatter,
    parameter_regex_search_string,
    parameter_set_reduction,
)


def test_variation_of_single_parameter():
    baseline = {"a": 0, "b": 0, "c": 0}
    variations = {"a": [1, 2], "c": [1]}

    parameter_settings = create_parameter_variations(variations, baseline, ["a", "b"])
    expected = [
        {"a": 1, "b": 0, "c": 0},
        {"a": 2, "b": 0, "c": 0},
    ]
    assert parameter_settings == expected


def test_variation_of_multiple_parameters():
    baseline = {"a": 1, "b": 2, "c": 3}
    variations = {"a": [3, 4, 5], "c": [1, 2]}

    parameter_settings = create_parameter_variations(variations, baseline)
    expected = [
        {"a": 3, "b": 2, "c": 1},
        {"a": 3, "b": 2, "c": 2},
        {"a": 4, "b": 2, "c": 1},
        {"a": 4, "b": 2, "c": 2},
        {"a": 5, "b": 2, "c": 1},
        {"a": 5, "b": 2, "c": 2},
    ]
    assert parameter_settings == expected


def test_parameter_dict_string_formatter():
    paramsettings = {"a": 3, "b": 2}
    expected = "a3e+00_b2e+00"
    assert parameter_dict_string_formatter(paramsettings, 0) == expected


def test_regex_param_search_generator_single_param():
    baseline = {"a": 1, "b": 2, "c": 3}
    free_params = ["a"]
    regex_string = parameter_regex_search_string(free_params, baseline, decimals=0)
    baseline_str = parameter_dict_string_formatter(baseline, decimals=0)
    assert re.match(regex_string, baseline_str) is not None


def test_param_projector():
    input_parameters = {"a": 2, "b": 3, "c": 4}
    baseline_parameters = {"a": 1, "b": 2, "c": 3}
    reduced = parameter_set_reduction(input_parameters, ["a", "b"], baseline_parameters)
    assert reduced == {"a": 2, "b": 3, "c": 3}
