FROM registry.datalab.tuwien.ac.at/jaas/base-notebook:latest

COPY --chown=1000:1000 . $HOME/microwave-remote-sensing
WORKDIR $HOME/microwave-remote-sensing

RUN pip install . --no-cache-dir
