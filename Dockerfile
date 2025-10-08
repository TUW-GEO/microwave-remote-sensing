FROM registry.datalab.tuwien.ac.at/jaas/base-notebook:latest

COPY --chown=1000:1000 . /app/microwave-remote-sensing
WORKDIR /app/microwave-remote-sensing

RUN pip install . --no-cache-dir