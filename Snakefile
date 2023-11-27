from twocomp.utils import float_string_formatter
from functools import partial
fsformat = partial(float_string_formatter, digits=2)

rule testrule:
    threads: 2
    shell:
        "python scripts/testscript.py {threads}"


rule baseline_models:
    input:
        "data/multidiffusion.hdf",
        "data/diffusion.hdf"

rule two_compartment_model:
    input:
        data="data/data.hdf",
        script="src/twocomp/multidiffusion.py",
    output:
        hdf="data/multidiffusion.hdf",
        total="data/multidiffusion_total.hdf",
    threads: 4
    shell:
        "python {input.script}"
        " --input {input.data}"
        " --output {output.hdf}"
        " --model 'fasttransfer'"
        " --visual"


rule single_compartment_model:
    input:
        data="data/data.hdf",
        script="src/twocomp/diffusion.py",
    output:
        hdf="data/diffusion.hdf",
        csv="data/diffusion.csv",
    threads: 4
    shell:
        "python {input.script}"
        " --input {input.data}"
        " --output {output.hdf}"
        " --model 'fasttransfer'"
        " --visual"

rule ecs_only_model:
    input:
        data="data/data.hdf",
        script="src/twocomp/diffusion.py",
    output:
        hdf="data/diffusion_ecs_only.hdf",
        csv="data/diffusion_ecs_only.csv",
    threads: 4
    shell:
        "python {input.script}"
        " --input {input.data}"
        " --output {output.hdf}"
        " --model 'singlecomp'"
        " --visual"


#############################
# PARAMETER VARIATION WORKFLOWS
#############################
D_e = 1.3e-4
D_p = D_e * 3
phi_e = 0.20
phi_p = 0.02
t_ep = 2.9e-2
t_pb = 0.2e-5
k_p = 3.7e-4
k_e = 1.0e-5

phi_tot = phi_e + phi_p

t_pb_max = 2.1e-5
k_e_max = 2.6e-5
D_p_list = [x * D_e for x in [3, 10, 100]]
t_ep_list = [5e-4, 3.1e-2]
t_pb_list = [scale * t_pb_max for scale in [0.0, 0.1, 1.0]]
k_e_list = [scale * k_e_max for scale in [1e-3, 1e-2, 1.0]]


rule singlecomp_varying_parameters_workflow:
    input:
        data="data/data.hdf",
    output:
        hdf="data/{modelname}/De{De}_Dp{Dp}_phie{phie}_phip{phip}_tep{tep}_tpb{tpb}_ke{ke}_kp{kp}.hdf",
    threads: 1  
    shell:
        " python scripts/singlecomp_wrapper.py"
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

rule singlecomp_varying_parameters:
    input:
        expand(
            "data/{modelname}/De{De}_Dp{Dp}_phie{phie}_phip{phip}_tep{tep}_tpb{tpb}_ke{ke}_kp{kp}.hdf",
            modelname=["singlecomp", "fasttransfer"],
            De = fsformat(1.3e-4),
            Dp = fsformat(D_e * 3),
            phie = fsformat(0.20),
            phip = fsformat(0.02),
            tep = fsformat(2.9e-2),
            tpb = fsformat(0.2e-5),
            kp = fsformat(3.7e-4),
            ke = fsformat(1.0e-5)
        )

