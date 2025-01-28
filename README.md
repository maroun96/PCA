# PCA

An implementation of principal component analysis using petsc4py and slepc4py (work in progress).

## Installation

First install miniconda by following the website's instruction: [link](https://docs.conda.io/projects/miniconda/en/latest/).

Verify that conda is properly installed on your system by typing: ```which conda```.

Create a conda environment and activate it: 

  - ```conda create -n <environment_name>```
  - ```conda activate <environment_name>```

The easiest way to install ```petsc4py``` and ```slepc4py``` is from the ```conda-forge``` channel:

- Add conda forge to your channels: ```conda config --add channels conda-forge``` ```conda config --set channel_priority strict```
- Install petsc4py and slepc4py: ```conda install petsc4py``` ```conda install slepc4py```

The petsc4py library needs python and mpi4py as dependencies, so they will be installed automatically.

## Running the script in parallel

MPI with 4 processes: ```mpiexec -n 4 python pod.py``` 

