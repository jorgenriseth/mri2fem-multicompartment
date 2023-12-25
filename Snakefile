import itertools 
from twocomp.utils import float_string_formatter, parameter_set_reduction
from functools import partial

rule baseline_models:
    input:
        "results/artificial_boundary/multidiffusion.hdf",
        "results/artificial_boundary/diffusion.hdf",
        "results/artificial_boundary/diffusion_ecs_only.hdf"

rule data_download:
    output:
        "data/data.hdf"
    shell:
        "wget -O data/data.zip 'https://www.dropbox.com/scl/fi/j6dfmk2bk3h0wvkx9ruzd/mri2fem-multicomp-data.zip?rlkey=xn4mli1otej1n8c6mnroa8adj&dl=1' &&"
        " unzip -d ./data ./data/data.zip &&"
        " rm data/data.zip"

###
# Baseline parameter models, Robin boundary condition, artificial SAS
###
rule two_compartment_model:
    input:
        data="data/data.hdf",
        script="src/twocomp/multidiffusion.py",
    output:
        hdf="results/artificial_boundary/multidiffusion.hdf",
        total="results/artificial_boundary/multidiffusion_total.hdf",
    threads: 4
    shell:
        "mpirun -n {threads}"
        " python {input.script}"
        " --input {input.data}"
        " --output {output.hdf}"
        " --output_total {output.total}"
        " --visual"


rule single_compartment_model:
    input:
        data="data/data.hdf",
        script="src/twocomp/diffusion.py",
    output:
        hdf="results/artificial_boundary/diffusion.hdf",
        csv="results/artificial_boundary/diffusion.csv",
    threads: 4
    shell:
        "mpirun -n {threads}"
        " python {input.script}"
        " --input {input.data}"
        " --output {output.hdf}"
        " --model 'fasttransfer'"
        " --visual"

rule ecs_only_model:
    input:
        data="data/data.hdf",
        script="src/twocomp/diffusion.py",
    output:
        hdf="results/artificial_boundary/diffusion_ecs_only.hdf",
        csv="results/artificial_boundary/diffusion_ecs_only.csv",
    threads: 4
    shell:
        "mpirun -n {threads}"
        " python {input.script}"
        " --input {input.data}"
        " --output {output.hdf}"
        " --model 'singlecomp'"
        " --visual"

###
# Baseline parameter models, Dirichlet boundary conditions, MRI-informed boundary
###
rule baseline_models_mri_boundary:
    input:
        "results/mri_boundary/multidiffusion.hdf",
        "results/mri_boundary/diffusion.hdf",
        "results/mri_boundary/diffusion_ecs_only.hdf"


rule two_compartment_model_mri_boundary:
    input:
        data="data/data.hdf",
        script="src/twocomp/multidiffusion.py",
    output:
        hdf="results/mri_boundary/multidiffusion.hdf",
        total="results/mri_boundary/multidiffusion_total.hdf",
    threads: 4
    shell:
        "mpirun -n {threads} python {input.script}"
        " --input {input.data}"
        " --output {output.hdf}"
        " --output_total {output.total}"
        " --ke 'inf'"
        " --kp 'inf'"
        " --visual"


rule single_compartment_model_mri_boundary:
    input:
        data="data/data.hdf",
        script="src/twocomp/diffusion.py",
    output:
        hdf="results/mri_boundary/diffusion.hdf",
        csv="results/mri_boundary/diffusion.csv",
    threads: 4
    shell:
        "mpirun -n {threads} python {input.script}"
        " --input {input.data}"
        " --output {output.hdf}"
        " --model 'fasttransfer'"
        " --k 'inf'"
        " --visual"

rule ecs_only_model_mri_boundary:
    input:
        data="data/data.hdf",
        script="src/twocomp/diffusion.py",
    output:
        hdf="results/mri_boundary/diffusion_ecs_only.hdf",
        csv="results/mri_boundary/diffusion_ecs_only.csv",
    threads: 4
    shell:
        "mpirun -n {threads} python {input.script}"
        " --input {input.data}"
        " --output {output.hdf}"
        " --model 'singlecomp'"
        " --k 'inf'"
        " --visual"


###
# Parameter variation workflows
### 
rule varying_parameters_workflow:
    input:
        data="data/data.hdf",
    output:
        hdf="results/{variationtype}/{modelname}/De{De}_Dp{Dp}_phie{phie}_phip{phip}_tep{tep}_tpb{tpb}_ke{ke}_kp{kp}.hdf",
    threads: 1  
    shell:
        " python scripts/simulation_runner.py"
        " --threads {threads}"
        " --input {input.data}"
        " --output {output.hdf}"
        " --model {wildcards.modelname}"
        " --De {wildcards.De}"
        " --Dp {wildcards.Dp}"
        " --phie {wildcards.phie}"
        " --phip {wildcards.phip}"
        " --tep {wildcards.tep}"
        " --tpb {wildcards.tpb}"
        " --ke {wildcards.ke}"
        " --kp {wildcards.kp}"


from twocomp.utils import parameter_dict_string_formatter, create_parameter_variations

base_params = {
    "De": 1.3e-4,
    "Dp": 3 * 1.3e-4,
    "phie": 0.20,
    "phip": 0.02,
    "tep": 2.9e-2,
    "tpb": 0.21e-5,
    "ke": 1.0e-5,
    "kp": 3.7e-4,
}

def single_param_variations(variation: dict[str, list[float]]) -> list[dict[str, float]]:
    d = create_parameter_variations(variation, base_params)
    psets = []
    for di in d:
        pstr = parameter_dict_string_formatter(di, decimals=2)
        if pstr not in psets:
            psets.append(pstr)
    return psets

D_p_list = [x * base_params["De"] for x in [3, 10, 100]]
phi_p_list = [0.01, 0.02, 0.04]
tep_list = [5e-4, 3.1e-2]
t_pb_max = 2.1e-5
tpb_list = [scale * t_pb_max for scale in [0.0, 0.1, 1.0]]
ke_max = 2.6e-5
ke_list = [scale * ke_max for scale in [1e-3, 1e-2, 1.0]]
kp_list = [0.9e-4, 3.7e-4, 7.4e-4]

rule single_param_variation_plot:
    input: 
        expand(
            "results/single_param/twocomp/{fname}.hdf",
            fname=single_param_variations({"Dp": D_p_list})
        ),
        expand(
            "results/single_param/twocomp/{fname}.hdf",
            fname=single_param_variations({"phip": phi_p_list})
        ),
        expand(
            "results/single_param/twocomp/{fname}.hdf",
            fname=single_param_variations({"tep": tep_list})
        ),
        expand(
            "results/single_param/twocomp/{fname}.hdf",
            fname=single_param_variations({"tpb": tpb_list})
        ),
        expand(
            "results/single_param/twocomp/{fname}.hdf",
            fname=single_param_variations({"ke": ke_list})
        ),
        expand(
            "results/single_param/twocomp/{fname}.hdf",
            fname=single_param_variations({"kp": kp_list})
        ),
    output:
        "figures/parameter-variations.pdf"
    shell:
        "python scripts/single_param_variation_plotter.py"
        

model_param_map = {
    "twocomp": ["De", "Dp", "phie", "phip", "tep", "tpb", "ke", "kp"],
    "fasttransfer": ["De", "Dp", "phie", "phip", "tpb", "ke", "kp"],
    "singlecomp": ["De", "phie", "ke", "tpb"],
}

def create_model_parameter_string_variations(
    variation, model_params, pase_params, decimals
):
    d = create_parameter_variations(variation, base_params)
    psets = []
    for di in d:
        pset = parameter_set_reduction(di, model_params, base_params)
        pstr = parameter_dict_string_formatter(pset, decimals)
        if pstr not in psets:
            psets.append(pstr)
    return psets

def create_variation(variation):
    return [
        (modelname, paramset)
        for modelname in model_param_map
        for paramset in create_model_parameter_string_variations(
            variation, model_param_map[modelname], base_params, 2
        )
    ]

def model_fname_list(variation):
    return [
        f"{modelname}/{fname}"
        for modelname, fname in create_variation(variation)
    ]

rule varying_tep_ke:
    input:
        expand(
            "results/compartmentalization/{model_fname}.hdf",
            model_fname = model_fname_list({
                "tep": [x * base_params["tep"] for x in [1e-4, 1e-3, 1e-2, 1.0]],
                "ke": [x * ke_max for x in [1e-1, 1, 1e1]],
            })
        )
    output:
        "figures/varying_tep_ke.pdf"
    shell:
        "python scripts/variation_transfer_boundary_parameter_plotter.py"



rule fenics2mri_workflow:
    input:
        referenceimage="data/concentration_0.mgz",
        timestampfile="data/timestamps.txt",
        datafile="data/data.hdf",
        simulationfile="results/mri_boundary/{funcname}.hdf",
    output:
        "results/mri/{funcname}_{idx}.nii.gz",
    shell:
        "python scripts/fenics2mri.py"
        " --simulationfile {input.simulationfile}"
        " --output {output}"
        " --referenceimage {input.referenceimage}"
        " --timestamps {input.timestampfile}"
        " --timeidx {wildcards.idx}"
        " --functionname 'total_concentration'"


rule fenics2mri:
    input:
        expand(
            "results/mri/{funcname}_{idx}.nii.gz",
            funcname="multidiffusion_total",
            idx=range(5),
        ),
    output:
        "figures/mri_concentration_row.pdf"
    shell:
        "python scripts/mri_concentration_images.py"


rule solute_quantification:
    input:
        "data/data.hdf",
    output:
        "results/quantities.csv",
    shell:
        "mpirun -n {threads}"
        " python scripts/solute_quantification.py"
        " --input {input} --funcname total_concentration --output {output}"


rule plots_all:
    input:
        "figures/sas-concentrations.pdf",
        "figures/parameter-variations.pdf",
        "figures/varying_tep_ke.pdf",
        "figures/mri_concentration_row.pdf",
        "figures/regionwise-models-data.pdf"

rule artificial_boundary_plot:
    input:
        csv = "results/quantities.csv",
        hdf = "data/data.hdf"
    output:
        "figures/sas-concentrations.pdf"
    shell:
        "python scripts/artificial_boundary_plots.py"

rule regionwise_quantities:
    input:
        "results/quantities.csv",
        "results/mri_boundary/multidiffusion.csv",
        "results/mri_boundary/diffusion.csv",
        "results/mri_boundary/diffusion_ecs_only.csv",
    output:
        "figures/regionwise-models-data.pdf"
    shell:
        "python scripts/region_quantities.py"

