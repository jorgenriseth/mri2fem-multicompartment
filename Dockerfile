FROM mambaorg/micromamba:latest

USER root
RUN apt-get update -y && apt-get install git wget zip -y
USER $MAMBA_USER


COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yaml 
RUN micromamba install -y -n base -f /tmp/environment.yaml && \
    micromamba clean --all --yes
WORKDIR /twocomp

COPY --chown=$MAMBA_USER:$MAMBA_USER LICENSE pyproject.toml README.md Snakefile ./
COPY --chown=$MAMBA_USER:$MAMBA_USER src/ src/
COPY --chown=$MAMBA_USER:$MAMBA_USER scripts/ scripts/
COPY --chown=$MAMBA_USER:$MAMBA_USER test/ test/
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
RUN pip install -e .

RUN mkdir data && \
    wget "https://www.dropbox.com/scl/fi/j6dfmk2bk3h0wvkx9ruzd/mri2fem-multicomp-data.zip?rlkey=xn4mli1otej1n8c6mnroa8adj&dl=1" \
    -O data/data.zip && \
    unzip -d ./data ./data/data.zip && \
    rm data/data.zip

EXPOSE 8080
