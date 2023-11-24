rule multidiffusion_model:
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
        " --visual"
