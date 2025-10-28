# UE Microwave Remote Sensing (120.030)

These are the hand-outs and exercises of the Master course Microwave Remote Sensing (120.030) at the TU Wien.

A guide for creating the notebooks and the environments in JupiterHub can be found in the file [`Index.qmd`](./Index.qmd).

# Generate Jupyter Conda environment and Jupyter Kernel from `yml`

To re-create the environment as a Jupyter kernel for execution of the notebooks, do the following:

- Open a Terminal from the same level as this Markdown README file.
- Type the following into the terminal.

```
make kernel
```

Select the kernel `microwave-remote-sensing`.

# Clean-up

To remove Jupyter checkpoints run:

```
make clean
```

In order to remove the Conda environments and Jupyter kernels run:

```
make teardown
```

# Developing

Use the `environment.yml` to setup a conda environment for developing the lecture notebooks. Commit notebooks without output for smaller file sizes and interactive teaching. For convenience use `nbstripout` to clean notebooks, like so:

```bash
pip install nbstripout
nbstripout **/*.ipynb

# Alternatively using uv (no installation necessary)
uvx nbstripout **/*.ipynb
```

> [!TIP]
> To use `uvx` the [uv] package manager needs to be installed.

Check you code for correct syntax as we want to show off good practices. You can use [ruff] to check your writing.

```bash
# To run the ruff linter
uvx ruff check --fix

# To run the ruff formatter
uvx ruff format
```

> [!IMPORTANT]
> Keep in mind, that the `homework` notebooks are automatically excluded from formatting with `ruff` and typechecking with `ty`, as defined in the `pyproject.toml` file.

The pre-commit hooks can be used to check whether outputs are empty. This can be achieved, like so:

```bash
pip install pre-commit
pre-commit install

# Alternatively using uv (no installation necessary)
uvx pre-commit install
```

To type check your code/notebook, and to make your code a little more static you might want to run:

```bash
uvx ty check
```

Alternatively a static type-checker like [mypy] can also be used, but that is more difficult to run on notebooks.

[ruff]: https://docs.astral.sh/ruff/
[uv]: https://docs.astral.sh/uv/
[mypy]: https://mypy.readthedocs.io/en/stable/index.html
