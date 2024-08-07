.ONESHELL:
SHELL = /bin/bash
.PHONY: help clean environment kernel post-render data

YML = $(wildcard *.yml)
REQ = $(basename $(notdir $(YML)))

CONDA_ENV_DIR := $(foreach i,$(REQ),$(shell conda info --base)/envs/$(i))
KERNEL_DIR := $(foreach i,$(REQ),$(shell jupyter --data-dir)/kernels/$(i))
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


help:
	@echo "Makefile for setting up environment, kernel, and pulling notebooks"
	@echo ""
	@echo "Usage:"
	@echo "  make environment  - Create Conda environments"
	@echo "  make kernel       - Create Conda environments and Jupyter kernels"
	@echo "  "
	@echo "  make teardown     - Remove Conda environments and Jupyter kernels"
	@echo "  make clean        - Removes ipynb_checkpoints"
	@echo "  make help         - Display this help message"

clean:
	rm --force --recursive .ipynb_checkpoints/

teardown:
	for i in $(REQ); do conda remove -n $$i --all -y ; jupyter kernelspec uninstall -y $$i ; done

$(CONDA_ENV_DIR):
	for i in $(YML); do conda env create -f $$i; done

environment: $(CONDA_ENV_DIR)
	@echo -e "conda environments are ready."

$(KERNEL_DIR): $(CONDA_ENV_DIR)
	pip install --upgrade pip
	pip install jupyter
	for i in $(REQ); do $(CONDA_ACTIVATE) $$i ; python -m ipykernel install --user --name $$i --display-name $$i ; conda deactivate; done

kernel: $(KERNEL_DIR)
	@echo -e "conda jupyter kernel is ready."