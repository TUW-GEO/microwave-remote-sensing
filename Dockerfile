FROM registry.datalab.tuwien.ac.at/jaas/base-notebook:latest

USER root

RUN apt-get update && apt-get install -yq --no-install-recommends \
    libgdal-dev \
    python3-dev \
    git \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

USER $NB_UID

RUN git clone --branch tuwel https://github.com/TUW-GEO/microwave-remote-sensing.git /home/jovyan/microwave-remote-sensing/
WORKDIR /home/jovyan/microwave-remote-sensing/

RUN uv pip install gdal=="$(gdal-config --version).*" --system && \
    uv pip install .[tuwel] --system
