
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <omp.h>
#include <cblas.h>
// #include <mkl.h>

#include "DenseMatrix.h"
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpParMat.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/SpDCCols.h"

namespace combblas{

  template class DenseMatrix<double>;


  template void processLines<double>(std::vector<std::string>& lines, int type, std::vector<double>& vals, int myrank);

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


  template void DenseMatrix<double>::ParallelReadDMM(const std::string & filename, bool onebased);

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
    MPI_Bcast(&type, 1, MPI_INT, 0, this->getCommWorld());
    MPI_Bcast(&symmetric, 1, MPI_INT, 0, this->getCommWorld());
    MPI_Bcast(&nrows, 1, MPIType<int64_t>(), 0, this->getCommWorld());
    MPI_Bcast(&ncols, 1, MPIType<int64_t>(), 0, this->getCommWorld());

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
    	  MPI_Bcast(&endofheader, 1, MPIType<MPI_Offset>(), 0, this->getCommWorld());
        // std::cout << "End of header is " << endofheader << " bytes" << std::endl;
        // std::string line;
        // std::getline(f, line);
        // int secondpos = ftell(f);
        // // calc the number of bytes in a line
        // int bytes_per_line = secondpos - fpos;
        // std::cout << "Bytes per line is " << bytes_per_line << std::endl;
        fclose(f);
    } else {
    	MPI_Bcast(&endofheader, 1, MPIType<MPI_Offset>(), 0, this->getCommWorld());  // receive the file loc at the end of header
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
    MPI_File_open(this->getCommWorld(), const_cast<char*>(filename.c_str()), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_fh);
	 
    typedef typename SpDCCols<int, NT>::LocalIT LIT;

    int number_of_elems_to_read = (end_fpos - fpos) / 15;
    std::vector<NT> vals;
    std::vector<std::string> lines;
    bool finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, true, lines, myrank);
    int64_t entriesread = lines.size();

    // std::cout << "Process " << myrank << " lines size " << entriesread << std::endl;
    processLines(lines, type, vals, myrank);   

    MPI_Barrier(this->getCommWorld());
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
    MPI_Reduce(&entriesread, &allentriesread, 1, MPIType<int64_t>(), MPI_SUM, 0, this->getCommWorld());
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
      localRows += nrows % grid_len;
    }
    if (procRow == grid_len - 1) {
      localCols += ncols % grid_len;
    }

    this->setLocalRows(localRows);
    this->setLocalCols(localCols);
    
    MPI_Barrier(this->getCommWorld());
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
    MPI_Barrier(this->getCommWorld());


  }

}