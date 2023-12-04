import sys
import csv
from itertools import islice
from itertools import tee

import numpy as np
import petsc4py
import petsc4py.PETSc as PETSc


def read_csv_parallel(filename, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)

        csv_reader_count, csv_reader_data = tee(csv_reader)
        row_count = sum(1 for _ in csv_reader_count)

        row_start, row_end = compute_row_indices(row_count, rank=rank, size=size)

        data = islice(csv_reader_data, row_start, row_end)

        str2float = lambda str_input: float(str_input)
        data_numeric = np.array([list(map(str2float, str_list)) for str_list in data])

    return data_numeric


def compute_row_indices(row_count, rank, size):
    row_start = 1 + rank*((row_count-1) // size)
    row_end = 1 + (rank+1)*((row_count-1) // size)
    if rank == size-1: row_end = row_count

    return row_start, row_end

if __name__ == "__main__":  
    comm = PETSc.COMM_WORLD
    petsc4py.init(args=sys.argv, comm=comm)

    rank = comm.Get_rank()
    size = comm.Get_size()
    
    local_data = read_csv_parallel(filename="../XYZ_dambreak_table_1.csv", comm=comm)

    print(f"rank {rank}: {local_data.shape}")
    
