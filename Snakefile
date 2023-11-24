rule multidiffusion_model:
    input:
        data="data/data.hdf",
        script="src/twocomp/multidiffusion.py",
    output:
        hdf="data/multidiffusion.hdf",
        total="data/multidiffusion_total.hdf",
    threads: 4

rule diffusion_model:
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
