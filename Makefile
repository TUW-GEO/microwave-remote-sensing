.ONESHELL:
SHELL = /bin/bash
.PHONY: help clean environment kernel teardown

YML = environment.yml
REQ = $(basename $(notdir $(YML)))
BASENAME = $(CURDIR)

CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
PREFIX = $(BASENAME)/.conda_envs
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
	rm --force --recursive .ipynb_checkpoints/ **/.ipynb_checkpoints/ _book/ \
		_freeze/ .quarto/

teardown:
	$(foreach f, $(REQ), \
		$(CONDA_ACTIVATE) $(f); \
		jupyter kernelspec uninstall -y $(f); \
		conda deactivate; \
		conda remove -n $(f) --all -y ; \
		conda deactivate; )

$(CONDA_ENV_DIR): $(YML)
	$(foreach f, $^, \
		conda env create --file $(f) \
			--prefix $(PREFIX)/$(basename $(notdir $(f))); )

environment: $(CONDA_ENV_DIR)
	@echo -e "conda environments are ready."
	pip install .
	@command -v pre-commit >/dev/null 2>&1 || pip install pre-commit
	python -m pre_commit install

$(KERNEL_DIR): $(CONDA_ENV_DIR)
	$(foreach f, $(REQ), \
		$(CONDA_ACTIVATE) $(f); \
		python -m ipykernel install --user --name $(f) --display-name $(f); \
		conda deactivate; )

kernel: $(KERNEL_DIR)
	@echo -e "conda jupyter kernel is ready."
	pip install .
	@command -v pre-commit >/dev/null 2>&1 || pip install pre-commit
	python -m pre_commit install
