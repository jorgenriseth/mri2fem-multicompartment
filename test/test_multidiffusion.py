import subprocess
from pathlib import Path


def test_twocomp_model(tmpdir):
    filedir = Path(tmpdir)
    cmd = (
        "python src/twocomp/multidiffusion.py"
        + " --input 'data/data.hdf'"
        + f" --output {filedir / 'out.hdf'}"
    )
    res = subprocess.run(cmd, shell=True)
    res.check_returncode()
    assert res.returncode == 0


def test_singlecomp_model(tmpdir):
    filedir = Path(tmpdir)
    cmd = (
        "python src/twocomp/diffusion.py"
        + " --input 'data/data.hdf'"
        + f" --output {filedir / 'out.hdf'}"
        + " --noquant"
    )
    res = subprocess.run(cmd, shell=True)
    res.check_returncode()
    assert res.returncode == 0
