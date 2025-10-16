.ONESHELL:
SHELL = /bin/bash
.PHONY: help clean teardown

CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
CONDA_ENV_DIR = ./.conda_envs/microwave-remote-sensing
KERNEL_DIR = $(shell jupyter --data-dir)/kernels/microwave-remote-sensing

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

teardown: $(CONDA_ENV_DIR)
	$(CONDA_ACTIVATE) $^
	jupyter kernelspec uninstall -y microwave-remote-sensing
	conda deactivate
	conda remove -p $^ --all -y
	conda deactivate

$(CONDA_ENV_DIR): ./environment.yml
	conda env create --file $^ --prefix $@ -y

environment: $(CONDA_ENV_DIR)
	@echo -e "conda environments are ready."

$(KERNEL_DIR): $(CONDA_ENV_DIR)
	$(CONDA_ACTIVATE) $^
	python -m ipykernel install --user --name microwave-remote-sensing --display-name microwave-remote-sensing
	pre-commit install
	conda deactivate

kernel: $(KERNEL_DIR)
	@echo -e "conda jupyter kernel is ready."
