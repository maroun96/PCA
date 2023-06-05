# POD

A small example that shows how to do the proper orthogonal decomposition of a matrix using petsc4py and slepc4py.

## Installation

The easiest way to install ```petsc4py``` and ```slepc4py``` is from the ```conda-forge``` channel:

- Add conda forge to your channels: ```conda config --add channels conda-forge``` ```conda config --set channel_priority strict```
- Install petsc4py and slepc4py: ```conda install petsc4py``` ```conda install slepc4py```

## Running the script in parallel

MPI with 4 processes: ```mpiexec -n 4 python pod.py``` 

