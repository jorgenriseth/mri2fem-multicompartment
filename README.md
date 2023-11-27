# Two-compartment diffusion model for tracer transport in brain tissue
This repository contains code for the chapter on two-compartment tracer transport models in the book MRI2FEM Volume 2.

## Setting up the python environment
We will be using conda as the python environment manager. To install the environment run:
```bash
conda env create -n twocomp -f environment.yml
conda activate twocomp
```

To install this repository as a package run
```bash
pip install -e .
```
`-e` signifies an editable install, in case you want to change the code yourself.

## Running simulations
We will be using [Snakemake](https://snakemake.readthedocs.io/) to execute our programs.
Snakemake is a workflow management system for defining and execute workflows defined 
by a set of input and output rules. It is governed by the human-friendly formatted `Snakefil`,
and defines workflows or "rules" based on input- and output-files, and a shell-command or 
python code needed to produce the output-files from the input files. 
This makes it easier to separate different parts of the workflow, as well as executing several
workflows in parallel.

```bash
snakemake baseline_models --cores 4
```
This will look into the `Snakefile`, find the rule 'baseline_models' and execute all rules needed
to produce the listed input files. The `--cores n` argument is a required argument telling snakemake
how many cores it has at its disposal. These cores can either be used to execute several rules 
in parallell, or to speed up rules that benefits from multiple cores (e.g. mpirun -n [n] for 
`fenics`-workflows).

Once completed, there should be `xdmf`-files in the directory `data/visual` representing
total concentration from both single-compartment (diffusion) and two-compartment concentration
(multidiffusion), as well as files for the fluid-concentrations in ECS and PVS.
The XDMF-files may be opened in e.g. [Paraview](https://www.paraview.org/) for inspecting the 
results in a 3D-viewer.

## Running with docker
Further description of the repository will be added. For now we stick to mentioning:

1. Download necessary data by using the following link
```bash
curl -o data/brain_mesh.h5 https://www.dropbox.com/s/43y6mvxugycua2f/brain_mesh.h5?dl=0
```
