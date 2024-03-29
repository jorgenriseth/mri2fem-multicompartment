# Two-compartment diffusion model for tracer transport in brain tissue
[![DOI](https://zenodo.org/badge/640939939.svg)](https://zenodo.org/doi/10.5281/zenodo.10820929)
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
by a set of input and output rules. It is governed by the human-friendly formatted `Snakefile`,
and defines workflows or "rules" based on input- and output-files, and a shell-command or 
python code needed to produce the output-files from the input files. 
This makes it easier to separate different parts of the workflow, as well as executing several
workflows in parallel
```bash
snakemake baseline_models --cores 4
```
This will look into the `Snakefile`, find the rule 'baseline_models' and execute all rules needed
to produce the listed input files. The `--cores N` argument is a required argument telling snakemake
how many cores it has at its disposal. These cores can either be used to execute several rules 
in parallell, or to speed up rules that benefits from multiple cores (e.g. mpirun -n [n] for 
`fenics`-workflows).

Once completed, there should be `xdmf`-files in the directory `data/visual` representing
total concentration from both single-compartment (diffusion) and two-compartment concentration
(multidiffusion), as well as files for the fluid-concentrations in ECS and PVS.
The XDMF-files may be opened in e.g. [Paraview](https://www.paraview.org/) for inspecting the 
results in a 3D-viewer.

The `Snakefile` describes workflows in terms of input and output files and can be consulted for specifics on how to run various scripts. 
By requesting a specific file, snakemake builds a DAG of files and figures out which workflows needs to be executed to create the requested file. 
By providing snakemake with the `-p` argument, it prints the necessary shell-command to execute the necessary workflows.
By additionally giving the argument `-n`, snakemake performs a "dry-run", only printing the workflows to be executed instead of running them.

To run all workflows necessary for creating the chapter plot figures, run the command
```bash
snakemake plots_all -c8 
```
This will
1. Download the input data.
2. Create mesh and convert mri-concentrations FEniCS-format functions in `hdf`-format.
3. Run all simulations with parameter variations needed for variuous plots.
4. Run scripts for reading the simulation output and creating figures from them. 

TODO: Add comment on `snakeconfig.yaml` and resolutions.

## Download input data 
*NB!: The data should not be publicly available until later, closer to the publication of the book. Preliminary access can be requested by reviewers and collaborators by sending a mail to jorgennr@simula.no *

Download the data by executing the command
```bash
snakemake data_download -c1
```


### Docker
It's also possible to run the examples using Docker. To build the docker image
```bash
docker build . -t twocomp
```
The run the container with a bash-shell using 
```bash 
docker run -it twocomp bash
```
Alternatively, run the container with the current diretory mounted to the volume
```bash
docker run -p 8080:8080 -v $(pwd):/twocomp -it twocomp
```
This could be useful to make simulation results persist, as the results are automatically output to your "local" directory. The port (`-p 8080:8080`) is exposed to enable running jupyter notebooks form within the container with
```bash 
jupyter notebook --port 8080 --ip 0.0.0.0
```

## Repository file structure
```
.
├── src                     # contains python code 
│   └── twocomp                 # twocom may be imported as a libary
│       ├── __init__.py
│       ├── diffusion.py        # code for the single-compartment diffusion models
│       ├── interpolator.py     # methods for temporal interpolation in FEM-represent form of data
│       ├── multidiffusion.py   # code for multi-compartment diffusion models
│       ├── parameters.py       # reference parameters and parameter processing
│       └── utils.py            # utility-functions used by other scripts

├── scripts                             # Collection of scripts for figure creation and runnign simulations
│   ├── artificial_boundary_plots.py    # Plots for 
│   ├── fenics2mri.py                   # Map concentrations from Fenics-format to MRI-images (nii.gz)
│   ├── mri_concentration_images.py     #  Create MRI-format images from simulation results.
│   ├── param_variation_plotter.py      # Plot single-parameter variations for two-comp model
│   ├── region_quantities.py            # Plot region-quantities for MRI-boundary models. 
│   ├── simulation_runner.py            # Common interface for running different simulation models.
│   ├── solute_quantification.py        # Compute  region solute content from hdf-files output as .csv

├── test                                # directory for automated tests (not very mature)
   └── test_parameter_iterators.py      # parameter estimation tests.

├── data
│   ├── data.hdf
│   └── timestamps.txt

├── results                         # results-directory, created by simuliations.
│   ├── artificial_boundary         # output HDF-files from artificial boundary baseline models
│   │   └── visual                  # output XDMF-files for paraview-visualization
│   ├── mri_boundary                # output-directory for MRI-data boundary baseline models 
│   │   └── visual                  # output XDMF-files for paraview-visualization
│   ├── compartmentalization        # Folder with parameter-cross-variations, organized by model
│   │   ├── fasttransfer
│   │   ├── singlecomp
│   │   └── twocomp
│   ├─ single_param                 # Output of single-parameter variation workflows.
│   └── mri                         # contains necessary MRI-images.

├── figures                         # Generated figures are stored in this directory   
├── Dockerfile  
├── environment.yml
├── LICENSE
├── pyproject.toml
├── README.md
├── Snakefile
```

### File structure `data.hdf`
```
data.hdf
├── domain 
│   ├── boundaries  # Dataset for a dolfin meshfunction containing boundary-labels
│   │   ├── coordinates {194848, 3}
│   │   ├── topology {1949822, 3} 
│   │   └── values {1949822}
│   ├── mesh  # Dolfin-compatible mesh
│   │   ├── cell_indices {920660}
│   │   ├── coordinates {194848, 3}
│   │   └── topology    {920660, 4}
│   └── subdomains  # Dataset for a dolfin meshfunction containing subdomain-labels
│       ├── cell_indices {920660}
│       ├── coordinates {194848, 3}
│       ├── topology {920660, 4}
│       └── values {920660} 
└── total_concentration  # Dataset containing MRI-concentrations interpolated onto mesh at different time-points
    ├── cell_dofs {368264
    ├── cells {920660}
    ├── vector_0 {194848}
    ├── vector_1 {194848}
    ├── vector_2 {194848}
    ├── vector_3 {194848}
    ├── vector_4 {194848}
    └── x_cell_dofs {920661}
```
