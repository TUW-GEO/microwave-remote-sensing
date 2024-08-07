# DataLab JupyterHub 

## Information for TU Wien Lecturers

This Makefile automatizes the process of cloning your GitHub repository to the student's JupyterLab Home folder. Before sharing this Makefile make sure that the account name, repository name, and branch name, are entered correctly. Place the Makefile in the Datalab Jupyter `shared` folder in order to make the notebooks available for students, or share the Makefile by any other means with the students.

## Information for TU Wien Students

To clone the lecturer's notebooks to your JupyterLab Home directory, do the following. 

- Open a Terminal from the same level as this Markdown README file.
- Type the following into the terminal.

```
make notebooks
```

Select the kernel with the equivalent name as the `.ipynb` notebook to execute the notebook. For example, `01_lecture.ipynb` requires the kernel `01_lecture` for execution of the code blocks.

To remove the notebooks as well as the Jupyter kernels, do the following.

- Open a Terminal from the same level as this Markdown README file.
- Type the following into the terminal.

```
make delete
```
