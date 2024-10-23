.ONESHELL:
SHELL = /bin/bash
.PHONY: help clean environment kernel teardown

YML = environment.yml
REQ = $(basename $(notdir $(YML)))
BASENAME = $(shell basename $(CURDIR))

CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
PREFIX = ${HOME}/$(BASENAME)/.conda_envs
CONDA_ENV_DIR := $(foreach i,$(REQ),$(PREFIX)/$(i))
KERNEL_DIR := $(foreach i,$(REQ),$(shell jupyter --data-dir)/kernels/$(i))

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
	$(foreach f, $(REQ), \
		jupyter kernelspec uninstall -y $(f); \
		conda remove -n $(f) --all -y ; \
		conda deactivate; )
	- conda config --remove envs_dirs $(PREFIX)

$(CONDA_ENV_DIR): $(YML)
	- conda config --prepend envs_dirs $(PREFIX)
	$(foreach f, $^, \
		conda env create --file $(f) \
			--prefix $(PREFIX)/$(basename $(notdir $(f))); )

environment: $(CONDA_ENV_DIR)
	@echo -e "conda environments are ready."

$(KERNEL_DIR): $(CONDA_ENV_DIR)
	$(foreach f, $(REQ), \
		$(CONDA_ACTIVATE) $(f); \
		python -m ipykernel install --user --name $(f) --display-name $(f); \
		conda deactivate; )

kernel: $(KERNEL_DIR)
	@echo -e "conda jupyter kernel is ready."