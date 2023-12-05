from mpi4py import MPI
import numpy as np

class commGrid:
    """
    a grid of processors
    reimplemented from 
    https://people.eecs.berkeley.edu/~aydin/CombBLAS/html/_comm_grid_8cpp_source.html 
    """
    def __init__(self, world: MPI.Comm, rows: int, cols: int) -> None:
        comm_world = MPI.Comm.Dup(world)
        self.comm_world = comm_world
                
        myrank = comm_world.Get_rank()
        n_proc = comm_world.Get_size()

        
        if rows == cols == 0:
            rows = cols = int(np.sqrt(n_proc))
            
            if rows * cols != n_proc:
                print("This version works on a square logical processor grid")
                MPI.COMM_WORLD.Abort(1) # TODO: define an error code?

            
        assert n_proc == rows * cols
        self.__rows = rows
        self.__cols = cols
        
        myprocol = myrank % cols
        mpprocrow = myrank // cols
        
        row_world = comm_world.Split(myprocol, mpprocrow)
        col_world = comm_world.Split(mpprocrow, myprocol)
        self.__row_world = row_world
        self.__col_world = col_world
        self.create_diag_world()
        
        row_rank =row_world.Get_rank()
        col_rank = col_world.Get_rank()
        assert row_rank == myprocol
        assert col_rank == mpprocrow
        # self.__row_rank = row_rank
        # self.__col_rank = col_rank

    def create_diag_world(self):
        """
        creates a communicator for the diagonal processors
        """
        if self.rows != self.cols:
            print("The grid is not square! Returning diagworld to everyone instaed of the diagonal")
            self.diag_world = self.comm_world
            return
        
        process_ranks = [i * self.cols + i for i in range(self.cols)]
            
        group = self.comm_world.Get_group()
        diag_group = group.Incl(process_ranks)
        MPI.Group.Free(group)
        
        self.diag_world = MPI.Comm.Create(self.comm_world, diag_group)
        MPI.Group.Free(diag_group)
        
    @property
    def row_world(self):
        return self.__row_world
    
    @property
    def col_world(self):
        return self.__col_world
    
    @property
    def rows(self):
        return self.__rows
    
    @property
    def cols(self):
        return self.__cols
    
    @property
    def size_vec(self):
        #TODO
        pass
    

class SpParMat:
    """
    currently a dummy placeholder
    """
    def __init__(self, rows: int, cols: int, values, grid) -> None:
        self.__localRows = rows
        self.__localCols = cols
        self.__values = values
        self.__commGrid = grid

class DenseMatrix:
    
    def __init__(self, rows: int, cols: int, values, grid=None) -> None:
        self.__localRows = rows
        self.__localCols = cols
        self.__values = values
        
        if grid is not None:
            self.__commGrid = grid
        else:
            self.__commGrid = commGrid(MPI.COMM_WORLD, rows, cols)
    
    @property
    def values(self):
        return self.__values
    
    @property
    def localRows(self):
        return self.__localRows
    
    @property
    def localCols(self):
        return self.__localCols
    
    @property
    def commGrid(self):
        return self.__commGrid
    
    
def __get_sending_rank_in_row(rank: int, diag_offset: int, cols: int):
    row_pos = rank/cols
    return row_pos * cols + (row_pos + diag_offset) % cols

def __get_recv_rank(rank: int, round: int, cols: int, size: int):
    recv_rank = rank - round * cols
    if recv_rank < 0:
        recv_rank = size + rank - (round * cols)
    return recv_rank
    
def __local_matrix_mult(A: DenseMatrix, B: SpParMat, clear_A: bool=False, clear_B: bool=False):
    # TODO: use a parallel matrix format
    B_elems = B.getSpSeq()
    nnz = B_elems.getnnz()
    cols_spars = B_elemes.getncol()
    rows_spars = B_elems.getnrow()
    local_cols = A.localCols()
    local_rows = A.localRows()
    values = A.values
    
    if (local_cols != rows_spars):
        raise Exception("Dimensions do not match")
    
    if nnz == 0:
        out_values = np.zeros((local_rows * local_rows))
        out = DenseMatrix(local_rows, local_cols, out_values,)
        return out
    
    B_dcsc  = B_elems.get_DCSC()
    out_values = np.zeros(local_cols * local_rows)
    
    for i in range(B.nzc):
        col = B_dcsc.jc[i]
        nnz_in_col = B_dcsc.jc[i+1] - B_dcsc.cp[i]
        
        for row in range(local_rows):
            sum = 0
            offset = row * local_cols
            for k in range(B_dcsc.cp[i], B_dcsc.cp[i] + nnz_in_col):
                sparse_row = B_dcsc.ir[k]
                elem = B_dcsc.numx[k]
                sum += values[offset + sparse_row] * elem
            out_values[offset + col] = sum
            
    out = DenseMatrix(local_rows, local_cols, out_values)
    return out
                

def fox(A: DenseMatrix, B:SpParMat)-> DenseMatrix:
    """
    fox-algorithm for matrix multiplication
    
    https://github.com/DakaiZhou/Fox-Algorithm
    """
    comm= MPI.COMM_WORLD
    comm_i = A.commGrid.world
    size = comm_i.Get_size()
    myrank = comm_i.Get_rank()
    
    row_dense = A.commGrid.rows
    col_dense = A.commGrid.cols
    
    row_sparse = B.commGrid.rows # TODO: use an appropritate SpMat format and do the correct calls
    col_sparse = B.commGrid.cols
    
    if myrank == 0:
        if row_dense != row_dense or col_dense != col_sparse or row_dense != col_dense:
            comm_i.Abort(1)
            
    B_elems = B.getSpSeq()
    results = []
    
    # round 0
    buffer_A = None # TODO: correctly initialize this vector
    size_vec = -1
    sending_rank = __get_sending_rank_in_row(myrank, 0, col_dense)
    if myrank == sending_rank:
        size_vec = A.values.size_vec # TODO: what is the size_vec of a grid?
        buffer_A = A.values
        
    A.commGrid.row_world.Bcast([size_vec, MPI.INT], root=sending_rank)#TODO: define row/col world
    buffer_A.resize(size_vec) # TODO: rewrite
    if isinstance(buffer_A, float):
        A.commGrid.row_world.Bcast([buffer_A, MPI.DOUBLE], root=sending_rank)
    
    A_tmp = DenseMatrix(row_dense, col_dense, buffer_A, A.commGrid)
    results.append(__local_matrix_mult(A_tmp, B))
    
    # other rounds
    buffer_B = None # TODO: correctly initialize this vector
    for round in range(1, col_dense):
        sending_rank = __get_sending_rank_in_row(myrank, round, col_dense)
        
        if myrank == sending_rank:
            size_vec = A.values.size_vec # TODO: what is the size_vec of a grid?
            buffer_A = A.values
            
        A.commGrid.row_world.Bcast([size_vec, MPI.INT], root=sending_rank) 
        buffer_A.resize(size_vec)
        if isinstance(buffer_A, float):
            A.commGrid.row_world.Bcast([buffer_A, MPI.DOUBLE], root=sending_rank)
            
        recv_rank = __get_recv_rank(myrank, round, col_dense, size)
        size_sparse = B_elems.getnnz()
        
        comm.isend([buffer_A, MPI.INT], dest=recv_rank, tag=0)
    
    
def PDGEMM(A: list, B: list, ROWS_A: int, COLS_B: int, COLS_A: int):
    """
    parallel dense GEMM
    """
    C = np.zeros((ROWS_A, COLS_B))

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    MATRIX_SIZE = ROWS_A * COLS_A
    
    rows_per_process = ROWS_A // size
    rows_per_process_remainder = ROWS_A % size
    
    my_rows = rows_per_process + (rank < rows_per_process_remainder)
    my_offset = rank * (rows_per_process + 1) * COLS_A \
        if rank < rows_per_process_remainder else \
        (rows_per_process_remainder * (rows_per_process + 1) * COLS_A) + \
        ((rank - rows_per_process_remainder) * rows_per_process * COLS_A)
        
    local_A = np.zeros(my_rows * COLS_A)
    local_C = np.zeros(my_rows * COLS_B)
    
    comm.Scatter(A, my_rows * COLS_A, MPI.DOUBLE, local_A, my_rows * COLS_A, MPI.DOUBLE, 0)
    # TODO: dgemm part
    
    comm.Gather(local_C, my_rows * COLS_B, MPI.DOUBLE, C, my_rows * COLS_B, MPI.DOUBLE, 0)
    
    # TODO: correctness check
    assert True
    return C