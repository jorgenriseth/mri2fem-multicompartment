FROM mambaorg/micromamba
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
WORKDIR /multidiffusion

EXPOSE 8080

ADD ./multidiffusion multidiffusion
ADD ./setup.py .

RUN pip install -e . 
