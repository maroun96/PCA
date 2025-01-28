import sys
import numpy as np 
import petsc4py
import petsc4py.PETSc as PETSc
import slepc4py.SLEPc as SLEPc

def initialize_petsc():
    """Initialize PETSc and return communicator and rank"""
    comm = PETSc.COMM_WORLD
    petsc4py.init(args=sys.argv, comm=comm)
    rank = comm.Get_rank()
    return comm, rank

def create_random_matrix(n, m, comm):
    """Create and assemble a PETSc matrix of size (n,m) with random values"""
    PETSc.Sys.Print("Assembling matrix", comm=comm)
    
    Ap = PETSc.Mat()
    Ap.create()
    Ap.setSizes((n,m))
    Ap.setUp()
    
    i_start, i_end = Ap.getOwnershipRange()
    process_values = np.random.random(size=(i_end-i_start, m))
    Ap.setValues(rows=range(i_start, i_end), cols=range(m), values=process_values)
    Ap.assemble()
    
    PETSc.Sys.Print("Assembly done", comm=comm)
    return Ap

def solve_eigenvalue_problem(matrix, comm):
    """Solve eigenvalue problem for given matrix using SLEPc"""
    PETSc.Sys.Print("Solving Eigenvalue Problem", comm=comm)
    
    EP = SLEPc.EPS()
    EP.create()
    EP.setOperators(matrix)
    EP.setType(SLEPc.EPS.Type.LAPACK)
    EP.setDimensions(nev=matrix.getSize()[0])
    EP.solve()
    
    eigenvalues = []
    for i in range(EP.getConverged()):
        eigenvalues.append(EP.getEigenvalue(i))
        PETSc.Sys.Print(f"iter: {i}", comm=comm)
    
    PETSc.Sys.Print("Eigenvalues Done", comm=comm)
    return eigenvalues, EP.getType()

def main():
    # Initialize PETSc
    comm, rank = initialize_petsc()
    
    # Problem dimensions
    n = 10000
    m = 100
    
    # Create random matrix
    Ap = create_random_matrix(n, m, comm)
    
    # Calculate B = Ap^T * Ap
    B = Ap.transposeMatMult(Ap)
    PETSc.Sys.Print("Multiplication Done", comm=comm)
    
    # Solve eigenvalue problem
    eigenvalues, solver_type = solve_eigenvalue_problem(B, comm)
    
    # Print results on rank 0
    if rank == 0:
        print(f'Eigenvalues (SLEPc {solver_type}): ', eigenvalues)

if __name__ == "__main__":
    main()