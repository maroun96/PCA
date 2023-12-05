#!/bin/bash

#SBATCH -J pod
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=imb,imb-resources,gpu


# useful informations to print
echo "#############################" 
echo "User:" $USER
echo "Date:" `date`
echo "Host:" `hostname`
echo "Directory:" `pwd`
echo "SLURM_JOBID:" $SLURM_JOBID
echo "SLURM_SUBMIT_DIR:" $SLURM_SUBMIT_DIR
echo "SLURM_JOB_NODELIST:" $SLURM_JOB_NODELIST
echo "#############################" 

#############################

mpirun -np 32 python loadsnap.py