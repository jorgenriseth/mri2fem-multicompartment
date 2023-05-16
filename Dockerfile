FROM continuumio/miniconda3

WORKDIR /multicompartment

COPY multicompartment.yml environment.yml
RUN conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda env create -f environment.yml && \
    conda activate multicompartment

EXPOSE 8080