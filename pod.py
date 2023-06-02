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

PETSc.Sys.Print("Assembling matrix", comm=comm)
Ap = PETSc.Mat()
Ap.create()
Ap.setSizes((n,m))
Ap.setUp()

i_start, i_end = Ap.getOwnershipRange()

process_values = np.random.random(size=(i_end-i_start, m))


Ap.setValues(rows = range(i_start, i_end), cols = range(m), values=process_values)
Ap.assemble()

PETSc.Sys.Print("Assembly done", comm=comm)

B = Ap.transposeMatMult(Ap)

PETSc.Sys.Print("Multiplication Done", comm=comm)

PETSc.Sys.Print("Solving Eigenvalue Problem", comm=comm)

EP = SLEPc.EPS()
EP.create()
EP.setOperators(B)
eptype = SLEPc.EPS.Type.LAPACK
EP.setType(eptype)
EP.setDimensions(nev = m)
EP.solve()

l_slepc = []
i=0
while i < EP.getConverged():
    l_slepc.append(EP.getEigenvalue(i))
    i += 1
    PETSc.Sys.Print(f"iter: {i}", comm=comm)

PETSc.Sys.Print("Eigenvalues Done", comm=comm)

if rank == 0:
    print(f'Eigenvalues (SLEPc {EP.getType()}): ', l_slepc)