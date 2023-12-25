from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--overlay_image", type=Path, required=True)
#     parser.add_argument("--reference_image", type=Path, required=True)
#     parser.add_argument(
#         "--orientation", choices=["saggital", "transversal", "coronal"], required=True
#     )
#     parser.add_argument("--slice_number", required=True)
#     parser = argparse.parse_args()
