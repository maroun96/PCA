import sys
import numpy as np 
import petsc4py
import petsc4py.PETSc as PETSc
import slepc4py.SLEPc as SLEPc

comm = PETSc.COMM_WORLD
petsc4py.init(args=sys.argv, comm=comm)

rank = comm.Get_rank()

n = 1000000
m = 1000

A = np.random.random(size=(n, m))

Ap = PETSc.Mat()
Ap.create()
Ap.setSizes(A.shape)
Ap.setUp()

i_start, i_end = Ap.getOwnershipRange()
process_values = A[i_start:i_end, :]


Ap.setValues(rows = range(i_start, i_end), cols = range(A.shape[1]), values=process_values)
Ap.assemble()

S = SLEPc.SVD()
S.create()
S.setOperator(Ap)
stype = SLEPc.SVD.Type.RANDOMIZED
S.setType(stype)
S.setDimensions(A.shape[0])
S.solve()

s_slepc = []
s_err = []
i=0
while i < S.getConverged():
    s_slepc.append(S.getValue(i))
    err = S.computeError(i)
    s_err.append(err)
    i += 1

if rank == 0:
    print(f'Singular values (SLEPc {S.getType()}): ', s_slepc)
    # print('errors:', s_err)