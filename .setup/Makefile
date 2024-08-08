.ONESHELL:
SHELL = /bin/bash
.PHONY: help

GIT_ACCOUNT = TUW-GEO
REPO_NAME = microwave-remote-sensing
GIT_BRANCH = main
GIT_REPO = https://github.com/$(GIT_ACCOUNT)/$(REPO_NAME).git
ROOT = ~/$(REPO_NAME)


help:
	@echo "Makefile for setting up environment, kernel, and pulling notebooks"
	@echo ""
	@echo "Usage:"
	@echo "  make notebooks    - Clone notebooks to JupyterLab, install Conda environments and Jupyter kernels"
	@echo "  "
	@echo "  make delete       - Remove Jupyter notebooks, Conda environments and Jupyter kernels"
	@echo "  make help         - Display this help message"

notebooks:
	git clone $(GIT_REPO) -b $(GIT_BRANCH) $(ROOT)
	cd $(ROOT) && make kernel
	@echo "Repository cloned to $(ROOT) and Jupyter kernels installed."

delete:
	@echo "Deleting all files in $(ROOT)..."
	cd $(ROOT) && make teardown
	rm -rf $(ROOT)
	@echo "$(ROOT) has been deleted."