from time import perf_counter
from math import floor

from mpi4py import MPI

import pandas as pd


def compute_col_indices(col_count, rank, size):
    col_start = floor(col_count*rank/size)
    col_length = floor(col_count*(rank+1)/size) - floor(col_count*rank/size)
    col_end = col_start + col_length
    
    return col_start, col_end

def read_csv_parallel(filename_list, col_count, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    col_start, col_end = compute_col_indices(col_count=col_count, rank=rank, size=size)

    local_filename_list = filename_list[col_start:col_end]

    df_list = []

    for filename in local_filename_list:
        df = pd.read_csv(filename, engine="pyarrow")
        df_list.append(df)
    
    return df_list



if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    col_count = 100

    filename_list = [f"/scratch/kmaroun/pod_files/snapshot{i}.csv" for i in range(1, 101)]

    time_start = perf_counter()
    _ = read_csv_parallel(filename_list=filename_list, col_count=100, comm=comm)
    time_end = perf_counter()

    print(f"rank {rank} - time elapsed: {time_end-time_start}")
    
