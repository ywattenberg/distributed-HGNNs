#ifndef _DENSE_MATRIX_H_
#define _DENSE_MATRIX_H_


#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <mpi.h>
#include <vector>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpParMat.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/SpDCCols.h"


namespace combblas{

template<class NT>
class DenseMatrix 
{
  public:
    void printLocalMatrix(){
      int myrank;
      MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
      std::cout << "local matrix A:" << std::endl;
      for (int i = 0; i < localRows * localCols; i++) {
          std::cout << "rank " << myrank << " local matrix A[" << i << "] = " << values->at(i) << std::endl;
      }
    }
    std::vector<NT>* getValues() {return values;}
    void setValues(std::vector<NT>* vals) {values = vals;}
    void push_back(NT val) {values.push_back(val);}
    int getLocalRows() {return localRows;}
    int getLocalCols() {return localCols;}
    void setLocalRows(int rows) {localRows = rows;}
    void setLocalCols(int cols) {localCols = cols;}
    std::shared_ptr<CommGrid> getCommGrid() {return commGrid;}
    auto getCommWorld() {return commGrid->GetWorld();}

    DenseMatrix<NT>(): values(nullptr), localRows(-1), localCols(-1), commGrid(nullptr) {
    }

    DenseMatrix<NT>(int rows, int cols, std::vector<NT> * values, std::shared_ptr<CommGrid> grid): values(values), localRows(rows), localCols(cols)
    {
      commGrid = grid;
    }

    DenseMatrix<NT>(int rows, int cols, std::shared_ptr<CommGrid> grid): localRows(rows), localCols(cols), values(nullptr)
    {
      commGrid = grid;
    }

    DenseMatrix<NT>(int rows, int cols, std::vector<NT> *values, MPI_Comm world): values(values), localRows(rows), localCols(cols)
    {
      commGrid.reset(new CommGrid(MPI_COMM_WORLD, rows, cols));
    }

    ~DenseMatrix<NT>()
    {
      if (values != nullptr) {
        delete values;
      }
    }

    void ParallelReadDMM (const std::string & filename, bool onebased);
    
    int getnrow() const
    {
      int totalrows = 0;  
      MPI_Allreduce( &localRows, &totalrows, 1, MPI_INT, MPI_SUM, commGrid->GetColWorld());
      return totalrows;  
    }

    int getncol() const
    {
      int totalcols = 0; 
      MPI_Allreduce( &localCols, &totalcols, 1, MPI_INT, MPI_SUM, commGrid->GetRowWorld());
      return totalcols;  
    }


    void GetPlaceInGlobalGrid(int &roffset, int &coffset){
      int total_rows = getnrow();
      int total_cols = getncol();

      int proc_rows = commGrid->GetGridRows();
      int proc_cols = commGrid->GetGridCols();

      int rows_perproc = total_rows / proc_rows;
	    int cols_perproc = total_cols / proc_cols;

      roffset = commGrid->GetRankInProcCol()*rows_perproc;
      coffset = commGrid->GetRankInProcRow()*cols_perproc;
    }
    
    
    void addBiasLocally(std::vector<NT>* bias)
    {
      int myrank;
      MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

      int localRows = getLocalRows();
      int localCols = getLocalCols();

      int roffset = 0;
      int coffset = 0;

      GetPlaceInGlobalGrid(roffset, coffset);

      std::cout << "i am rank " << myrank << " with offset " << roffset << " " << coffset << std::endl;
      
      for (int i = 0; i < localRows; i++){
        for (int j = 0; j < localCols; j++){
          values->at(i*localCols + j) += bias->at(coffset + j);
        }
      }
    }

  private:
    int localRows;
    int localCols;
    std::vector<NT> *values;
    std::shared_ptr<CommGrid> commGrid; 
};


template<typename IT, typename NT>	
static void BCastMatrixDense(MPI_Comm & comm1d, std::vector<NT> * values, std::vector<IT> essentials, int sendingRank)
{
  int myrank;
  MPI_Comm_rank(comm1d, &myrank);


  MPI_Bcast(essentials.data(), essentials.size(), MPIType<IT>(), sendingRank, comm1d);

  if (myrank != sendingRank){
    values->resize(essentials[0]);
  }

  MPI_Bcast(values->data(), essentials[0], MPIType<NT>(), sendingRank, comm1d); 
}

template<typename SR, typename IT, typename NT, typename DER>
void blockDenseSparse(size_t dense_rows, size_t dense_cols, std::vector<NT>* dense_A, DER* sparse_B, std::vector<NT> * outValues){
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  
  IT nnz = sparse_B->getnnz();
  IT cols_spars = sparse_B->getncol();
  IT rows_spars = sparse_B->getnrow();

  if (dense_cols != rows_spars) {
    throw std::invalid_argument( "DIMENSIONS DON'T MATCH" );        
  }

  if (nnz == 0){
    return;
  }

  Dcsc<IT, NT>* Bdcsc = sparse_B->GetDCSC();
  for(size_t i = 0; i < dense_rows; i++){
    for (size_t j = 0; j < Bdcsc->nzc; j++){
      IT col = Bdcsc->jc[j];
      size_t nnzInCol = Bdcsc->cp[j+1] - Bdcsc->cp[j];
      for(size_t k =0; k < nnzInCol; k++){
        IT sparseRow = Bdcsc->ir[Bdcsc->cp[j]+ k];
        NT elem = Bdcsc->numx[Bdcsc->cp[j]+ k];
        outValues->at(i * cols_spars + col) += SR::multiply(dense_A->at(i * dense_cols + sparseRow), elem);
      }
    }
  }
}
// TODO: Parallelize 
template<typename SR, typename NT>
inline DenseMatrix<NT> DenseDenseAdd(DenseMatrix<NT> &A, DenseMatrix<NT> &B){
  size_t rows = A.getLocalRows(); 
  size_t cols = A.getLocalCols();
  if (rows != B.getLocalRows() || cols != B.getLocalCols()) {
    throw std::invalid_argument( "DIMENSIONS DON'T MATCH" );        
  }
  std::vector<NT> out = new std::vector<NT>(rows * cols);
  auto dense_A = A.getValues();
  auto dense_B = B.getValues();
  std::transform(dense_A->begin(), dense_A->end(), dense_B->begin(), out->begin(), SR::add);
  return DenseMatrix<NT>(rows, cols, out, A.getCommGrid()); 
}

// TODO: Write Function once we have bias and parallize
template<typename SR, typename IT, typename NT>
inline DenseMatrix<NT> DenseVecAdd(DenseMatrix<NT> &A, std::vector<NT> &B){
  int rows = A.getLocalRows();
  int cols = A.getLocalCols();
  auto commGrid = A.getCommGrid();

  //Find location of local rows in overall grid
  int rowDense = commGrid->GetGridRows();
  int colDense = commGrid->GetGridCols();
  int rankAinRow = commGrid->GetRankInProcRow();

  std::vector<NT>* out = new std::vector<NT>(rows * cols);
  // Add corresponding entries of B to A (offset by global pos)
  for (int i = 0; i < rows; i++){
    for (int j = 0; j < cols; j++){
      out->at(i * cols + j) = SR::add(A.getValues()->at(i * cols + j), B.at(i + rankAinRow * rows));
    }
  }

  return DenseMatrix<NT>(rows, cols, out, commGrid);
}


// TODO: Parallelize 
template<typename SR, typename NT>
inline DenseMatrix<NT> DenseReLU(DenseMatrix<NT> &A){
  size_t rows = A.getLocalRows(); 
  size_t cols = A.getLocalCols();
  std::vector<NT>* out = new std::vector<NT>(rows * cols);
  auto dense_A = A.getValues();
  for(int i = 0; i < rows * cols; i++){
    out->at(i) = dense_A->at(i) > 0 ? dense_A->at(i) : static_cast<NT>(0.0);
  }
  return DenseMatrix<NT>(rows, cols, out, A.getCommGrid());
}

// TODO: Parallelize 
template<typename SR, typename NT>
inline DenseMatrix<NT> DerivativeDenseReLU(DenseMatrix<NT> &A){
  size_t rows = A.getLocalRows(); 
  size_t cols = A.getLocalCols();
  std::vector<NT>* out = new std::vector<NT>(rows * cols);
  auto dense_A = A.getValues();
  for(int i = 0; i < rows * cols; i++){
    out->at(i) = dense_A->at(i) > 0 ?  static_cast<NT>(1.0) : static_cast<NT>(0.0);
  }
  return DenseMatrix<NT>(rows, cols, out, A.getCommGrid());
}


template<typename SR, typename IT, typename NT, typename DER>
DenseMatrix<NT> DenseSpMult(DenseMatrix<NT> &A, SpParMat<IT, NT, DER> &B) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    int rowDense = A.getCommGrid()->GetGridRows();
    int colDense = A.getCommGrid()->GetGridCols();

    int stages, dummy;
    std::shared_ptr<CommGrid> GridC = ProductGrid((A.getCommGrid()).get(), (B.getcommgrid()).get(), stages, dummy, dummy);		
    

    IT ** BRecvSizes = SpHelper::allocate2D<IT>(DER::esscount, stages);
    SpParHelper::GetSetSizes( *(B.getSpSeq()), BRecvSizes, (B.getcommgrid())->GetColWorld());


    std::vector<NT> * bufferA = new std::vector<NT>();
    std::vector<IT> essentialsA(3); // saves rows, cols and total number of elements of block
    std::vector<IT> ess = std::vector<IT>();
    std::vector<NT> * out;
    DER * bufferB;

    int rankAinRow = A.getCommGrid()->GetRankInProcRow();
    int rankAinCol = A.getCommGrid()->GetRankInProcCol();
    int rankBinCol = B.getcommgrid()->GetRankInProcCol();

    int denseLocalRows = A.getLocalRows();
    int denseLocalCols = A.getLocalCols();

    int sparseLocalRows = B.getlocalrows();
    int sparseLocalCols = B.getlocalcols();

    std::vector<NT> * localOut = new std::vector<NT>(denseLocalRows * sparseLocalCols);
    

    //other stages:
    for (int i = 0; i < stages; i++){
      int sendingRank = i;
      
      if (rankAinRow == sendingRank){
        bufferA = A.getValues();

        essentialsA[1] = A.getLocalRows();
        essentialsA[2] = A.getLocalCols();
        essentialsA[0] = essentialsA[1] * essentialsA[2];
      }

      BCastMatrixDense(GridC->GetRowWorld(), bufferA, essentialsA, sendingRank);

      if(rankBinCol == sendingRank)
      {
        bufferB = B.getSpSeq();
      }
      else
      {
        ess.resize(DER::esscount);		
        for(int j=0; j< DER::esscount; ++j)	
        {
          ess[j] = BRecvSizes[j][i];
        }	
        bufferB = new DER();
      }
      
      SpParHelper::BCastMatrix<IT, NT, DER>(GridC->GetColWorld(), *bufferB, ess, sendingRank);

      blockDenseSparse<SR, IT, NT, DER>(denseLocalRows, denseLocalCols, bufferA, bufferB, localOut);

    }

    return DenseMatrix<NT>(denseLocalRows, sparseLocalCols, localOut, GridC);
}

template<typename SR, typename IT, typename NT, typename DER>
void blockSparseDense(size_t dense_rows, size_t dense_cols, DER* sparse_B, std::vector<NT>* dense_A, std::vector<NT> * outValues){
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  
  IT nnz = sparse_B->getnnz();
  IT cols_spars = sparse_B->getncol();
  IT rows_spars = sparse_B->getnrow();

  if (cols_spars != dense_rows) {
    throw std::invalid_argument( "DIMENSIONS DON'T MATCH" );        
  }

  if (nnz == 0){
    return;
  }

  Dcsc<IT, NT>* Bdcsc = sparse_B->GetDCSC();

  for (size_t i = 0; i < Bdcsc->nzc; i++){
    IT Col = Bdcsc->jc[i];
    size_t nnzInCol = Bdcsc->cp[i+1] - Bdcsc->cp[i];
    for (size_t j = 0; j < nnzInCol; j++){
      IT sparseRow = Bdcsc->ir[Bdcsc->cp[i]+ j];
      NT elem = Bdcsc->numx[Bdcsc->cp[i]+ j];

      for (size_t k = 0; k < dense_cols; k++){
        outValues->at(sparseRow * dense_cols + k) += SR::multiply(elem, dense_A->at(Col * dense_cols + k));
      }
    }
  }
}


// computes B*A, where B is a sparse matrix and A a dense matrix
template<typename SR, typename IT, typename NT, typename DER>
DenseMatrix<NT> SpDenseMult(SpParMat<IT, NT, DER> &B, DenseMatrix<NT> &A) 
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  int rowDense = A.getCommGrid()->GetGridRows();
  int colDense = A.getCommGrid()->GetGridCols();

  int stages, dummy;
  std::shared_ptr<CommGrid> GridC = ProductGrid((B.getcommgrid()).get(), (A.getCommGrid()).get(), stages, dummy, dummy);		
  

  IT ** BRecvSizes = SpHelper::allocate2D<IT>(DER::esscount, stages);
  SpParHelper::GetSetSizes( *(B.getSpSeq()), BRecvSizes, (B.getcommgrid())->GetRowWorld());


  std::vector<NT> * bufferA = new std::vector<NT>();
  std::vector<IT> essentialsA(3); // saves rows, cols and total number of elements of block
  std::vector<IT> ess = std::vector<IT>();
  std::vector<NT> * out;
  DER * bufferB;

  int rankAinRow = A.getCommGrid()->GetRankInProcRow();
  int rankAinCol = A.getCommGrid()->GetRankInProcCol();
  int rankBinRow = B.getcommgrid()->GetRankInProcRow();
  int rankBinCol = B.getcommgrid()->GetRankInProcCol();

  int denseLocalRows = A.getLocalRows();
  int denseLocalCols = A.getLocalCols();

  int sparseLocalRows = B.getlocalrows();
  int sparseLocalCols = B.getlocalcols();

  std::vector<NT> * localOut = new std::vector<NT>(sparseLocalRows * denseLocalCols);
  
  //other stages:
  for (int i = 0; i < stages; i++){    
    int sendingRank = i;
    
    if (rankAinCol == sendingRank){
      bufferA = A.getValues();

      essentialsA[1] = A.getLocalRows();
      essentialsA[2] = A.getLocalCols();
      essentialsA[0] = essentialsA[1] * essentialsA[2];
    }

    BCastMatrixDense(GridC->GetColWorld(), bufferA, essentialsA, sendingRank);

    if(rankBinRow == sendingRank)
    {
      bufferB = B.getSpSeq();
    }
    else
    {
      ess.resize(DER::esscount);		
      for(int j=0; j< DER::esscount; ++j)	
      {
        ess[j] = BRecvSizes[j][i];
      }	
      bufferB = new DER();
    }
    
    SpParHelper::BCastMatrix<IT, NT, DER>(GridC->GetRowWorld(), *bufferB, ess, sendingRank);

    blockSparseDense<SR, IT, NT, DER>(denseLocalRows, denseLocalCols, bufferB, bufferA, localOut);
  }

  return DenseMatrix<NT>(denseLocalRows, sparseLocalCols, localOut, GridC);

}

// helper function for ParallelReadDMM
template <class NT>
void processLines(std::vector<std::string>& lines, int type, std::vector<NT>& vals, int myrank) {
   if(type == 0)   // real
    {
        double vv;
        for (auto itr=lines.begin(); itr != lines.end(); ++itr)
        {
            // string::c_str() -> Returns a pointer to an array that contains a null-terminated sequence of characters (i.e., a C-string)
            sscanf(itr->c_str(), "%lg", &vv);
            // print read values
            // std::cout << "process " << myrank << " read value " << vv << std::endl;
            vals.push_back(vv);
        }
    }
    else if(type == 1) // integer
    {
        int64_t vv;
        for (auto itr=lines.begin(); itr != lines.end(); ++itr)
        {
            sscanf(itr->c_str(), "%lld", &vv);
            vals.push_back(vv);
        }
    }
    else
    {
        std::cout << "COMBBLAS: Unrecognized matrix market scalar type" << std::endl;
    }
    lines.clear();
}

inline int getOwner(int nrows, int ncols, int row, int col, int grid_len) {
    int rows_per_proc = nrows / grid_len;
    int cols_per_proc = ncols / grid_len;
    int rowid = row / rows_per_proc;
    rowid = std::min(rowid, grid_len - 1);
    int colid = col / cols_per_proc;
    colid = std::min(colid, grid_len - 1);
    int rank = rowid * grid_len + colid;
    // std::cout << "row " << row << " col " << col << " is owned by rank " << rank << std::endl;
    return rank;
}

// ! Handles all sorts of orderings, even duplicates (what happens to them is determined by BinOp)
// ! Requires proper matrix market banner at the moment
// ! Replaces ReadDistribute for properly load balanced input in matrix market format
template <class NT>
void DenseMatrix< NT >::ParallelReadDMM(const std::string & filename, bool onebased) {
    int32_t type = -1;
    int32_t symmetric = 0;
    int32_t nrows, ncols;
    int32_t linesread = 0;
    
    FILE *f;
    int myrank = commGrid->GetRank();
    int nprocs = commGrid->GetSize();

    // get info about file
    if(myrank == 0) { // only the root processor reads the file{
        MM_typecode matcode;
        if ((f = fopen(filename.c_str(), "r")) == NULL)
        {
            printf("COMBBLAS: Matrix-market file %s can not be found\n", filename.c_str());
            std::cout << "COMBBLAS: Matrix-market file " << filename << " can not be found" << std::endl;
            SpParHelper::Print("COMBBLAS: Matrix-market file " + filename + " can not be found");
            MPI_Abort(MPI_COMM_WORLD, NOFILE);
        }
        if (mm_read_banner(f, &matcode) != 0)
        {
            printf("Could not process Matrix Market banner.\n");
            exit(1);
        }
        linesread++;
        
        if (mm_is_complex(matcode))
        {
            printf("Sorry, this application does not support complex types");
            printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        }
        else if(mm_is_real(matcode))
        {
            std::cout << "Matrix is Float" << std::endl;
            type = 0;
        }
        else if(mm_is_integer(matcode))
        {
            std::cout << "Matrix is Integer" << std::endl;
            type = 1;
        }
        else if(mm_is_pattern(matcode))
        {
            std::cout << "Matrix is Boolean" << std::endl;
            type = 2;
        }
        if(mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
        {
            std::cout << "Matrix is symmetric" << std::endl;
            symmetric = 1;
        }
        int ret_code;
        if ((ret_code = mm_read_mtx_array_size(f, &nrows, &ncols)) !=0)  // ABAB: mm_read_mtx_crd_size made 64-bit friendly
            exit(1);
    
        std::cout << "Total number of rows, columns expected across all processors is " << nrows << ", " << ncols << std::endl;
    }

    // broadcast matrix info
    MPI_Bcast(&type, 1, MPI_INT, 0, getCommWorld());
    MPI_Bcast(&symmetric, 1, MPI_INT, 0, getCommWorld());
    MPI_Bcast(&nrows, 1, MPIType<int64_t>(), 0, getCommWorld());
    MPI_Bcast(&ncols, 1, MPIType<int64_t>(), 0, getCommWorld());

    int rows_per_proc = (nrows / nprocs);
    int row_start = myrank * rows_per_proc;

    // std::cout << "nrows: " << nrows << std::endl;
    // std::cout << "rows per proc: " << rows_per_proc << std::endl;
    // std::cout << "row start: " << row_start << std::endl;
    // Use fseek again to go backwards two bytes and check that byte with fgetc
    struct stat st;     // get file size
    if (stat(filename.c_str(), &st) == -1) {
        std::cout << "COMBBLAS: Could not determine file size" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, NOFILE);
    }

    // distribute file offsets to all processes
    int64_t file_size = st.st_size;
    MPI_Offset fpos, end_fpos, endofheader;
    if(commGrid->GetRank() == 0) { // the offset needs to be for this rank
        std::cout << "File is " << file_size << " bytes" << std::endl;
        fpos = ftell(f);
        endofheader =  fpos;
    	  MPI_Bcast(&endofheader, 1, MPIType<MPI_Offset>(), 0, getCommWorld());
        // std::cout << "End of header is " << endofheader << " bytes" << std::endl;
        // std::string line;
        // std::getline(f, line);
        // int secondpos = ftell(f);
        // // calc the number of bytes in a line
        // int bytes_per_line = secondpos - fpos;
        // std::cout << "Bytes per line is " << bytes_per_line << std::endl;
        fclose(f);
    } else {
    	MPI_Bcast(&endofheader, 1, MPIType<MPI_Offset>(), 0, getCommWorld());  // receive the file loc at the end of header
	    fpos = endofheader + myrank * (file_size-endofheader) / nprocs;
      // give each process a number of rows to read
      // fpos = endofheader + myrank * rows_per_proc * ncols * 15;
    }

    if(myrank != (nprocs-1)) {
      end_fpos = endofheader + (myrank + 1) * (file_size-endofheader) / nprocs;
      // end_fpos = endofheader + (myrank + 1) * rows_per_proc * ncols * 15;
    } else {
      end_fpos = file_size;
    }
    std::cout << "Process " << myrank << " will read from " << fpos << " to " << end_fpos << std::endl;
    MPI_File mpi_fh;
    MPI_File_open(getCommWorld(), const_cast<char*>(filename.c_str()), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_fh);
	 
    typedef typename SpDCCols<int, NT>::LocalIT LIT;

    int number_of_elems_to_read = (end_fpos - fpos) / 15;
    std::vector<NT> vals;
    std::vector<std::string> lines;
    bool finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, true, lines, myrank);
    int64_t entriesread = lines.size();

    // std::cout << "Process " << myrank << " lines size " << entriesread << std::endl;
    processLines(lines, type, vals, myrank);   

    MPI_Barrier(getCommWorld());
    if (myrank == 0) {
      std::cout << "First batch finished" << std::endl;
    }

    while(!finished) {
        // std::cout << "Process " << myrank << " reading from " << fpos << std::endl;
        finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, false, lines, myrank);
        entriesread += lines.size();
        // SpHelper::ProcessLines(rows, cols, vals, lines, symmetric, type, onebased);
        processLines(lines, type, vals, myrank);
    }

    int64_t allentriesread;
    MPI_Reduce(&entriesread, &allentriesread, 1, MPIType<int64_t>(), MPI_SUM, 0, getCommWorld());
// #ifdef COMBBLAS_DEBUG
    if(myrank == 0)
        std::cout << "Reading finished. Total number of entries read across all processors is " << allentriesread << std::endl;
// #endif

    // SpParMat<int, NT, Sp>()::SparseCommon(data, allentriesread, nrows, ncols, PlusTimesSRxing<NT,NT>());
    MPI_File_close(&mpi_fh);
    // std::vector<NT>().swap(&vals);

    int rows_read = entriesread / ncols;
    // std::cout << "process " << myrank << " read " << entriesread << " entries" << std::endl;

    int grid_len = std::sqrt(nprocs);
    int localRows = nrows / grid_len;
    int localCols = ncols / grid_len;

    int procCol = commGrid->GetRankInProcCol();
    int procRow = commGrid->GetRankInProcRow();

    if (procCol == grid_len - 1) {
      localCols += ncols % grid_len;
    }
    if (procRow == grid_len - 1) {
      localRows += nrows % grid_len;
    }

    this->setLocalRows(localRows);
    this->setLocalCols(localCols);
    
    MPI_Barrier(getCommWorld());
    // std::cout << "process " << myrank << " local rows: " << localRows << " " << " local cols: " << localCols << std::endl;
    std::vector<std::vector<NT>> data = std::vector<std::vector<NT>>(nprocs);
    // std::cout << "process " << myrank << " vals size: " << vals.size() << std::endl;

    for (int i = 0; i < rows_read; i++) {
      for (int j = 0; j < ncols; j++) {
        int recvRank = getOwner(nrows, ncols, (myrank * rows_per_proc) + i, j, grid_len);
        data[recvRank].push_back(vals.at(i * ncols + j));
      }
    }

    // send data to correct processors
    int * sendcnt = new int[nprocs];
    int * recvcnt = new int[nprocs];
    for(int i=0; i<nprocs; ++i)
      sendcnt[i] = data[i].size();	// sizes are all the same

    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetWorld()); // share the counts
    int * sdispls = new int[nprocs]();
    int * rdispls = new int[nprocs]();
    std::partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
    std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
    int totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<int>(0));
    int totsent = std::accumulate(sendcnt,sendcnt+nprocs, static_cast<int>(0));	
    
    assert((totsent < std::numeric_limits<int>::max()));	
    assert((totrecv < std::numeric_limits<int>::max()));


    int locsize = vals.size();
  	NT * senddata = new NT[locsize];	// re-used for both rows and columns
    for(int i=0; i<nprocs; ++i)
    {
      std::copy(data[i].begin(), data[i].end(), senddata+sdispls[i]);
      data[i].clear();	// clear memory
      data[i].shrink_to_fit();
    }
    MPI_Datatype MPI_NT;
    MPI_Type_contiguous(sizeof(NT), MPI_CHAR, &MPI_NT);
    MPI_Type_commit(&MPI_NT);

    NT * recvdata = new NT[totrecv];	
    MPI_Alltoallv(senddata, sendcnt, sdispls, MPI_NT, recvdata, recvcnt, rdispls, MPI_NT, commGrid->GetWorld());

    DeleteAll(senddata, sendcnt, recvcnt, sdispls, rdispls);
    MPI_Type_free(&MPI_NT);
    std::vector<NT>* tmp = new std::vector<NT>(recvdata, recvdata + totrecv);
    // print received data
    for (int i = 0; i < totrecv; i++) {
      // std::cout << "process " << myrank << " received value " << recvdata[i] << std::endl;
      tmp->push_back(recvdata[i]);
    }
    this->setValues(tmp);
    MPI_Barrier(getCommWorld());


}

template<typename SR, typename NT>
void blockDenseDense(size_t rowsA, size_t colsA, size_t rowsB, size_t colsB, std::vector<NT>* dense_A, std::vector<NT>* dense_B, std::vector<NT>* outValues)
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if (colsA != rowsB) {
    throw std::invalid_argument("DIMENSIONS DON'T MATCH");
  }

  for (size_t i = 0; i < rowsA; i++){
    for (size_t j = 0; j < colsB; j++){
      for (size_t k = 0; k < colsA; k++){
        // // std::cout << "index out: " << i * colsB + j << std::endl;
        // // std::cout << "index in: " << i * colsA + k << std::endl;
        // std::cout << "index in 2: " << k* colsB + j << std::endl;
        // std::cout << "size: " << dense_B->size() << std::endl;

        outValues->at(i * colsB + j) += SR::multiply(dense_A->at(i * colsA + k), dense_B->at(k* colsB + j));
      }
    }
  }
}


template<typename SR, typename NT>
DenseMatrix<NT> DenseDenseMult(DenseMatrix<NT> &A, DenseMatrix<NT> &B)
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int stages, dummy;
  std::shared_ptr<CommGrid> GridC = ProductGrid((A.getCommGrid()).get(), (B.getCommGrid()).get(), stages, dummy, dummy);		

  std::vector<NT> * bufferA = new std::vector<NT>();
  std::vector<int> essentialsA(3);
  std::vector<NT> * bufferB = new std::vector<NT>();
  std::vector<int> essentialsB(3);

  int rankAinRow = A.getCommGrid()->GetRankInProcRow();
  int rankBinCol = A.getCommGrid()->GetRankInProcCol();

  int localRowsA = A.getLocalRows();
  int localColsA = A.getLocalCols();
  int localRowsB = B.getLocalRows();
  int localColsB = B.getLocalCols();

  std::vector<NT> * localOut = new std::vector<NT>(localRowsA * localColsB);

  for (int i = 0; i < stages; i++){
    int sendingRank = i;
    
    if (rankAinRow == sendingRank){
      bufferA = A.getValues();

      essentialsA[1] = localRowsA;
      essentialsA[2] = localColsA;
      essentialsA[0] = essentialsA[1] * essentialsA[2];
    }

    BCastMatrixDense(GridC->GetRowWorld(), bufferA, essentialsA, sendingRank);

      
    if (rankBinCol == sendingRank){
      bufferB = B.getValues();

      essentialsB[1] = localRowsB;
      essentialsB[2] = localColsB;
      essentialsB[0] = essentialsB[1] * essentialsB[2];
    }

    BCastMatrixDense(GridC->GetColWorld(), bufferB, essentialsB, sendingRank);

    blockDenseDense<SR, NT>(localRowsA, localColsA, localRowsB, localColsB, bufferA, bufferB, localOut);
  
  }
    
  return DenseMatrix<NT>(localRowsA, localColsB, localOut, GridC);
}
}


#endif