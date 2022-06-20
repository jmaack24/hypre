/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"
#include "par_ilu.h"
#include "seq_mv.hpp"

#ifndef HYPRE_ILDL_DEBUG
#define HYPRE_ILDL_DEBUG
#endif
#undef HYPRE_ILDL_DEBUG

HYPRE_Int
hypre_ILUSetupILDLTDevice(hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Real *tol, HYPRE_Int *perm,
                          HYPRE_Int *qperm, HYPRE_Int n, HYPRE_Int nLU,
                          hypre_GpuMatData * matL_des, hypre_GpuMatData * matU_des,
                          hypre_CsrsvData ** matBLU_csrsvdata_ptr, hypre_CsrsvData ** matSLU_csrsvdata_ptr,
                          hypre_CSRMatrix **BLUptr, hypre_ParCSRMatrix **matSptr, hypre_CSRMatrix **Eptr,
                          hypre_CSRMatrix **Fptr,
                          HYPRE_Int **A_fake_diag_ip,
                          HYPRE_Int tri_solve, char * mmfilename)
{
#ifdef HYPRE_ILDL_DEBUG
   hypre_printf("%s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
   fflush(NULL);
#endif

   /* GPU-accelerated ILU0 with cusparse */
   HYPRE_Int               i, j, k1, k2, k3, col;

   /* communication stuffs for S */
   MPI_Comm                comm                 = hypre_ParCSRMatrixComm(A);

   HYPRE_Int               my_id, num_procs;
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   hypre_ParCSRCommPkg     *comm_pkg;
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int               num_sends, begin, end;
   HYPRE_BigInt            *send_buf            = NULL;
   HYPRE_Int               *rperm               = NULL;
   HYPRE_Int               *rqperm              = NULL;

   //hypre_ParCSRMatrix      *Apq                 = NULL;
   hypre_CSRMatrix         *A_diag              = NULL;
   hypre_ParCSRMatrix      *ALDL                 = NULL;

   hypre_ParCSRMatrix      *matS                = NULL;
   //hypre_CSRMatrix         *A_diag              = NULL;
   HYPRE_Int               *A_fake_diag_i       = NULL;
   hypre_CSRMatrix         *A_offd              = NULL;
   HYPRE_Int               *A_offd_i            = NULL;
   HYPRE_Int               *A_offd_j            = NULL;
   HYPRE_Real              *A_offd_data         = NULL;
   hypre_CSRMatrix         *SLU                 = NULL;

   /* opaque pointers to vendor library data structs */
   hypre_CsrsvData         *matBLU_csrsvdata    = NULL;
   hypre_CsrsvData         *matSLU_csrsvdata    = NULL;

   /* variables for matS */
   HYPRE_Int               m                    = n - nLU;
   HYPRE_Int               nI                   = nLU;//use default
   HYPRE_Int               e                    = 0;
   HYPRE_Int               m_e                  = m;
   HYPRE_BigInt            total_rows;
   HYPRE_BigInt            col_starts[2];
   HYPRE_Int               *S_diag_i            = NULL;
   HYPRE_Int               S_diag_nnz;
   hypre_CSRMatrix         *S_offd              = NULL;
   HYPRE_Int               *S_offd_i            = NULL;
   HYPRE_Int               *S_offd_j            = NULL;
   HYPRE_Real              *S_offd_data         = NULL;
   HYPRE_BigInt            *S_offd_colmap       = NULL;
   HYPRE_Int               S_offd_nnz;
   HYPRE_Int               S_offd_ncols;

   /* set data slots */
   A_offd                                       = hypre_ParCSRMatrixOffd(A);
   A_offd_i                                     = hypre_CSRMatrixI(A_offd);
   A_offd_j                                     = hypre_CSRMatrixJ(A_offd);
   A_offd_data                                  = hypre_CSRMatrixData(A_offd);

   /* unfortunately we need to build the reverse permutation array */
   rperm                                        = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
   rqperm                                       = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
#ifdef HYPRE_ILDL_DEBUG
   hypre_printf("%s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
   fflush(NULL);
#endif

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   if (n > 0)
   {
      HYPRE_THRUST_CALL( sequence,
                         thrust::make_permutation_iterator(rperm, perm),
                         thrust::make_permutation_iterator(rperm+n, perm+n),
                         0 );
      HYPRE_THRUST_CALL( sequence,
                         thrust::make_permutation_iterator(rqperm, qperm),
                         thrust::make_permutation_iterator(rqperm+n, qperm+n),
                         0 );
   }
#else
   // not sure if this works
   for (i = 0; i < n; i++)
   {
      rperm[perm[i]] = i;
      rqperm[qperm[i]] = i;
   }
#endif

#ifdef HYPRE_ILDL_DEBUG
   hypre_printf("%s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
   fflush(NULL);
#endif

   /* Only call ILU when we really have a matrix on this processor */
   if (n > 0)
   {
      /* Copy diagonal matrix into a new place with permutation
       * That is, A_diag = A_diag(perm,qperm);
       */
      hypre_CSRMatrixApplyRowColPermutation(hypre_ParCSRMatrixDiag(A), perm, rqperm, &A_diag);

      //hypre_ParILURAPReorder( A, perm, rqperm, &Apq);

      /* Apply ILU factorization to the entile A_diag */
#ifdef HYPRE_ILDL_DEBUG
      hypre_printf("%s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
      fflush(NULL);
#endif

      MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
      hypre_ILUSetupILDLTNoPivot(A_diag, lfil, tol, NULL, NULL, n, n, &ALDL, comm, mmfilename);
      hypre_CSRMatrixDestroy(A_diag);
      /* | L \ U (B) L^{-1}F  |
       * | EU^{-1}   L \ U (S)|
       * Extract submatrix L_B U_B, L_S U_S, EU_B^{-1}, L_B^{-1}F
       * Note that in this function after ILU, all rows are sorted
       * in a way different than HYPRE. Diagonal is not listed in the front
       */

      /* No need to call this */
      //hypre_ILUSetupLDUtoCusparse( parL, parD, parU, &ALU);

      A_diag = hypre_ParCSRMatrixDiag(ALDL);

      hypre_ParILUDeviceILUExtractEBFC(A_diag, nLU, BLUptr, &SLU, Eptr, Fptr);

      if (A_diag)
      {
         hypre_CSRMatrixDestroy(A_diag);
      }

   }
   else
   {
      *BLUptr = NULL;
      *Eptr = NULL;
      *Fptr = NULL;
      SLU = NULL;
   }

   /* create B */
   /* only analyse when nacessary */
   if ( nLU > 0 )
   {
      /* Analysis of BILU */
      if (tri_solve)
      {
         HYPRE_ILUSetupCusparseCSRILU0SetupSolve(*BLUptr, matL_des, matU_des,
                                                 &matBLU_csrsvdata);
      }
   }

   HYPRE_BigInt big_m = (HYPRE_BigInt)m;
   hypre_MPI_Allreduce(&big_m, &total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
   /* only form when total_rows > 0 */
   if ( total_rows > 0 )
   {
      /* now create S */
      /* need to get new column start */
      {
         HYPRE_BigInt global_start;
         hypre_MPI_Scan( &big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
         col_starts[0] = global_start - m;
         col_starts[1] = global_start;
      }

      A_fake_diag_i = hypre_CTAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
      if (SLU)
      {
         /* Analysis of SILU */
          if (tri_solve)
          {
             HYPRE_ILUSetupCusparseCSRILU0SetupSolve(SLU, matL_des, matU_des,
                                                     &matSLU_csrsvdata);
          }
      }
      else
      {
         SLU = hypre_CSRMatrixCreate(0, 0, 0);
         hypre_CSRMatrixInitialize(SLU);
      }
      S_diag_i = hypre_CSRMatrixI(SLU);
      S_diag_nnz = S_diag_i[m];
      /* Build ParCSRMatrix matS
       * For example when np == 3 the new matrix takes the following form
       * |IS_1 E_12 E_13|
       * |E_21 IS_2 E_22| = S
       * |E_31 E_32 IS_3|
       * In which IS_i is the cusparse ILU factorization of S_i in one matrix
       * */

      /* We did nothing to A_offd, so all the data kept, just reorder them
       * The create function takes comm, global num rows/cols,
       *    row/col start, num cols offd, nnz diag, nnz offd
       */
      S_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);
      S_offd_ncols = hypre_CSRMatrixNumCols(A_offd);

      matS = hypre_ParCSRMatrixCreate( comm,
                                       total_rows,
                                       total_rows,
                                       col_starts,
                                       col_starts,
                                       S_offd_ncols,
                                       S_diag_nnz,
                                       S_offd_nnz);

      /* first put diagonal data in */
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matS));
      hypre_ParCSRMatrixDiag(matS) = SLU;

      /* now start to construct offdiag of S */
      S_offd = hypre_ParCSRMatrixOffd(matS);
      S_offd_i = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_colmap = hypre_CTAlloc(HYPRE_BigInt, S_offd_ncols, HYPRE_MEMORY_HOST);

      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i[i] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col + 1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i + 1 + e] = k3;
      }

      /* give I, J, DATA to S_offd */
      hypre_CSRMatrixI(S_offd) = S_offd_i;
      hypre_CSRMatrixJ(S_offd) = S_offd_j;
      hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      /* setup comm_pkg if not yet built */
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      }
      /* get total num of send */
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
      end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      send_buf = hypre_TAlloc(HYPRE_BigInt, end - begin, HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for (i = begin; i < end; i++)
      {
         send_buf[i - begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)] - nLU + col_starts[0];
      }

      /* main communication */
      comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf, S_offd_colmap);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;

      hypre_ILUSortOffdColmap(matS);

      /* free */
      hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
   }
   else {
       /** need to clean this up here potentially if its empty */
       if (hypre_CSRMatrixNumRows(SLU)==0 && hypre_CSRMatrixNumNonzeros(SLU)==0) {
           hypre_CSRMatrixDestroy( SLU );
           SLU = NULL;
       }
   }
   /* end of forming S */

   *matSptr       = matS;
   *matBLU_csrsvdata_ptr    = matBLU_csrsvdata;
   *matSLU_csrsvdata_ptr    = matSLU_csrsvdata;
   *A_fake_diag_ip = A_fake_diag_i;

   /* Destroy the bridge after acrossing the river */
   hypre_TFree(rperm, HYPRE_MEMORY_DEVICE);
   hypre_TFree(rqperm, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

HYPRE_Int hypre_LCSC_RowKtimesDenseVector(HYPRE_Int n, HYPRE_Int k, HYPRE_Int * csc_rows, HYPRE_Int * csc_col_offsets, HYPRE_Real * csc_data, HYPRE_Real * x, HYPRE_Real * y)
{
   HYPRE_Int i=0, j=0;
   for (i=0; i<k; ++i) {
      for (j=csc_col_offsets[i]; j<csc_col_offsets[i+1]; ++j) {
          if (csc_rows[j]==k) {
            y[i] = csc_data[j]*x[i];
         }
      }
   }
   return hypre_error_flag;
}

HYPRE_Int hypre_LCSCtimesDenseVector(HYPRE_Int n, HYPRE_Int k, HYPRE_Int * csc_rows, HYPRE_Int * csc_col_offsets, HYPRE_Real * csc_data, HYPRE_Real * x, HYPRE_Real * y)
{
    HYPRE_Int i=0, j=0, ii=0;
    for (i=0; i<k; ++i) {
        HYPRE_Real xx = x[i];
        for (j=csc_col_offsets[i]; j<csc_col_offsets[i+1]; ++j) {
            ii = csc_rows[j];
            if (ii<k) continue;
            y[ii] += csc_data[j]*xx;
        }
    }
   return hypre_error_flag;
}


HYPRE_Int hypre_LCSCSparseToDenseColumnVector(HYPRE_Int n, HYPRE_Int k, HYPRE_Int * csc_col_offsets, HYPRE_Int * csc_rows, HYPRE_Real * csc_data, HYPRE_Real * x)
{
   HYPRE_Int j=0;
   for (j=csc_col_offsets[k]; j<csc_col_offsets[k+1]; ++j) {
      if (csc_rows[j]>=k)
         x[csc_rows[j]] = csc_data[j];
   }
   return hypre_error_flag;
}

HYPRE_Int hypre_DenseVectorDropEntriesAfterK(HYPRE_Int n, HYPRE_Int k,  HYPRE_Real * x, HYPRE_Real tol)
{
   /* compute norm below the diagonal. Not sure if I should include unit diagonal or not */
   HYPRE_Int j=0;
   HYPRE_Int col_k_nnz=1;
   if (tol>0.0) {
      HYPRE_Real mag = 0.0;
      for (j=k; j<n; ++j)
         mag += x[j]*x[j];
      mag = sqrt(mag);
      for (j=k+1; j<n; ++j) {
          if (std::abs(x[j])<tol*mag)
            x[j]=0.0;
         else {
            col_k_nnz++;
         }
      }
   } else {
      for (j=k+1; j<n; ++j) {
         if (x[j]!=0.0)
            col_k_nnz++;
      }
   }
   return col_k_nnz;
}

HYPRE_Int print_L(HYPRE_Int n, HYPRE_Int k, HYPRE_Int * csc_col_offsets, HYPRE_Int * csc_rows, HYPRE_Real * csc_data, HYPRE_Real * diag)
{
   for (int j=0; j<k; ++j) {
       for (int l=csc_col_offsets[j]; l<csc_col_offsets[j+1]; ++l) {
           hypre_printf("\t(%d,%d) : %1.5g\n",csc_rows[l],j,csc_data[l]);
       }
   }
   hypre_printf("\n");
   for (int j=0; j<k; ++j) {
      hypre_printf("\t(%d,%d) : %1.5g\n",j,j,diag[j]);
   }
   fflush(NULL);
   return hypre_error_flag;
}

HYPRE_Int
hypre_ILUSetupILDLTNoPivot(hypre_CSRMatrix *A_diag, HYPRE_Int fill_factor, HYPRE_Real *tol,
                           HYPRE_Int *permp, HYPRE_Int *qpermp, HYPRE_Int nLU, HYPRE_Int nI, hypre_ParCSRMatrix **LDLptr,
                           MPI_Comm comm, char * mmfilename)
{
#ifdef HYPRE_USING_CUDA
   cudaEvent_t start, stop;
   float time;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
#endif

   HYPRE_Int i=0, j=0, k=0;

    //hypre_CSRMatrix          *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real              *A_data              = hypre_CSRMatrixData(A_diag);
   HYPRE_Int               *A_i                 = hypre_CSRMatrixI(A_diag);
   HYPRE_Int               *A_j                 = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int               n                    = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int               m                    = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int               nnz_A                = hypre_CSRMatrixNumNonzeros(A_diag);

   hypre_assert(n == m);

   hypre_CSRMatrix          *AT_diag;
   HYPRE_Real              *d_Acsc_data;
   HYPRE_Int               *d_Acsc_col_offsets;
   HYPRE_Int               *d_Acsc_rows;

   cusparseMatDescr_t descr = hypre_CSRMatrixGPUMatDescr(A_diag);

   /* Make sure the array is row sorted first */
   hypre_SortCSRCusparse(n, m, nnz_A, descr, A_i, A_j, A_data);

   /* transpose the matrix puts A in CSC form */
   hypre_CSRMatrixTransposeDevice(A_diag, &AT_diag, 1);

   d_Acsc_data = hypre_CSRMatrixData(AT_diag);
   d_Acsc_col_offsets = hypre_CSRMatrixI(AT_diag);
   d_Acsc_rows = hypre_CSRMatrixJ(AT_diag);

   /* Move the data back to the host */
   HYPRE_Int * Acsc_col_offsets = hypre_CTAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_HOST);
   HYPRE_Int * Acsc_rows = hypre_CTAlloc(HYPRE_Int, nnz_A, HYPRE_MEMORY_HOST);
   HYPRE_Real * Acsc_data = hypre_CTAlloc(HYPRE_Real, nnz_A, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(Acsc_col_offsets, d_Acsc_col_offsets, HYPRE_Int, n+1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(Acsc_rows, d_Acsc_rows, HYPRE_Int, nnz_A, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(Acsc_data, d_Acsc_data, HYPRE_Real, nnz_A, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixDestroy(AT_diag);

   /* set the initial capacity */
   HYPRE_Int capacity = nnz_A;
   HYPRE_Int * Lcsc_col_offsets = hypre_CTAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_HOST);
   HYPRE_Int * Lcsc_rows = hypre_CTAlloc(HYPRE_Int, capacity, HYPRE_MEMORY_HOST);
   HYPRE_Real * Lcsc_data = hypre_CTAlloc(HYPRE_Real, capacity, HYPRE_MEMORY_HOST);

   HYPRE_Int * Lcsc_col_count = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   hypre_Memset(Lcsc_col_count, 0, sizeof(HYPRE_Int)*n, HYPRE_MEMORY_HOST);

   HYPRE_Real * D_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
   hypre_Memset(D_data, 0, sizeof(HYPRE_Real)*n, HYPRE_MEMORY_HOST);

   HYPRE_Int lfil = fill_factor*nnz_A/m;

   /* Crout Grief ILDL
      L is going to generated in CSC form. Once finished, we'll convert to CSR
    */

#ifdef HYPRE_USING_CUDA
   cudaEventRecord(start, 0);
   HYPRE_Int lastk=0;
#endif

   hypre_printf("%s %s %d : drop tolerance=%1.5g\n",__FILE__,__FUNCTION__,__LINE__,tol[0]);
   hypre_printf("%s %s %d : lfil=%d\n",__FILE__,__FUNCTION__,__LINE__,lfil);

   /*************************************************************************/
   /* Initialize the first column                                           */
   /*************************************************************************/

   /* initialize the diagonal */
   D_data[0] = Acsc_data[0];

   /* initialize the first column L by dividing each column value by the diagonal */
   Lcsc_data[0] = 1.0;
   Lcsc_rows[0] = 0;

   if (tol[0]>0.0) {
       /* compute norm below the diagonal. Not sure if I should include unit diagonal or not */
       Lcsc_col_count[0]=1;
       HYPRE_Real mag=0.0;
       for (i=0; i<Acsc_col_offsets[1]; ++i) {
           mag += (Acsc_data[i]/D_data[0])*(Acsc_data[i]/D_data[0]);
       }
       mag = std::sqrt(mag);
       HYPRE_Int cnt=1;
       for (i=1; i<Acsc_col_offsets[1]; ++i) {
           HYPRE_Real temp = Acsc_data[i]/D_data[0];
           if (std::abs(temp)>=tol[0]*mag) {
#ifdef HYPRE_ILDL_DEBUG
               hypre_printf("%s %s %d : col=0, row=%d, temp=%1.5g, mag=%1.5g\n",__FILE__,__FUNCTION__,__LINE__,i,temp,mag);
#endif
               Lcsc_rows[cnt] = Acsc_rows[i];
               Lcsc_data[cnt++] = temp;
               Lcsc_col_count[0]++;
           }
       }
   }
   else
   {
       for (i=1; i<Acsc_col_offsets[1]; ++i) {
           Lcsc_rows[i] = Acsc_rows[i];
           Lcsc_data[i] = Acsc_data[i]/D_data[0];
       }
       Lcsc_col_count[0] = Acsc_col_offsets[1];
   }

   /* exclusive scan */
   Lcsc_col_offsets[0]=0;
   for (i=1; i<n+1; ++i)
       Lcsc_col_offsets[i] = Lcsc_col_count[i-1]+Lcsc_col_offsets[i-1];

   HYPRE_Int Lcsc_nnz = Lcsc_col_offsets[1];

#ifdef HYPRE_ILDL_DEBUG
   hypre_printf("%s %s %d : Lcsc_nnz=%d\n",__FILE__,__FUNCTION__,__LINE__,Lcsc_nnz);
   fflush(NULL);
#endif

   HYPRE_Real * temp1 = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
   HYPRE_Real * temp2 = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
   HYPRE_Real * avect = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
   HYPRE_Real * temp3 = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);

   /*************************************************************************/
   /* Loop over the remaining columns                                       */
   /*************************************************************************/

   for (k=1; k<n; ++k) {
       /* force these to 0 at each iteration */
       hypre_Memset(temp1, 0, sizeof(HYPRE_Real)*n, HYPRE_MEMORY_HOST);
       hypre_Memset(temp2, 0, sizeof(HYPRE_Real)*n, HYPRE_MEMORY_HOST);
       hypre_Memset(avect, 0, sizeof(HYPRE_Real)*n, HYPRE_MEMORY_HOST);
       hypre_Memset(temp3, 0, sizeof(HYPRE_Real)*n, HYPRE_MEMORY_HOST);

       /************/
       /* Update L */
       /************/

       /* 1) elementwise : temp1 =  L[k,:k] .* Diag[:k] */
       hypre_LCSC_RowKtimesDenseVector(n, k, Lcsc_rows, Lcsc_col_offsets, Lcsc_data, D_data, temp1);

       /* 2) normal spmv : avect =  L[k:n,:k] * temp1 */
       hypre_LCSCtimesDenseVector(n, k, Lcsc_rows, Lcsc_col_offsets, Lcsc_data, temp1, avect);

       /* 3) L[k:n,k] = A[k:n,k] - avect
 
         A is sparse csc, avect has full storage but it is sparsely populated
          need to compute collisions first, then rellocate storage
        */

       /* Take sparse column to dense below the diagonal */
       hypre_LCSCSparseToDenseColumnVector(n, k, Acsc_col_offsets, Acsc_rows, Acsc_data, temp2);

       /* scale by the diagonal */
       D_data[k] = temp2[k]-avect[k];
       for (j=k; j<n; ++j) {
           HYPRE_Real t = temp2[j]-avect[j];
           if (t!=0) {
               temp3[j]=t/D_data[k];
           }
       }

#ifdef HYPRE_ILDL_DEBUG
       printf("\n");
       for (i=0;i<n;++i) hypre_printf("%s %s %d : col=%d, row=%d, temp1=%1.16g  avect=%1.16g  Acsc=%1.16g  temp3=%1.16g, diag=%1.16g\n",__FILE__,__FUNCTION__,__LINE__,i,k,temp1[i],avect[i],temp2[i],temp3[i],D_data[k]);
       printf("\n");
       fflush(NULL);
#endif

       /* 4) Drop tolerance */
       HYPRE_Int col_k_nnz = hypre_DenseVectorDropEntriesAfterK(n, k,  temp3, tol[0]);

#ifdef HYPRE_ILDL_DEBUG
       hypre_printf("%s %s %d : col_k_nnz=%d\n",__FILE__,__FUNCTION__,__LINE__,col_k_nnz);
       fflush(NULL);
#endif

       /* Move data to short buffer */
       HYPRE_Real * temp4_data = hypre_CTAlloc(HYPRE_Real, col_k_nnz, HYPRE_MEMORY_HOST);
       HYPRE_Int * temp4_rows = hypre_CTAlloc(HYPRE_Int, col_k_nnz, HYPRE_MEMORY_HOST);
       i=0;
       for (j=k; j<n; ++j)
       {
          if (temp3[j]!=0.0)
          {
              temp4_rows[i]=j;
              temp4_data[i++]=temp3[j];
          }
       }

       if (lfil>0) {

           if (col_k_nnz > lfil + 1) {

	     /* Zero all but 'lfil' largest elements below the diagonal */

	     if (temp4_rows[0] != k)
	       {
		 hypre_printf("%s %s %d : Value above the diagonal -- col=%d row=%d value=%g\n",__FILE__,__FUNCTION__,__LINE__, k, temp4_rows[0], temp4_data[0]);
	       }

	     /* Make sure diagonal value is not removed by skipping it -- should
	      always be the first element in temp4_data and temp4_row */
	     thrust::stable_sort_by_key(thrust::host,
					temp4_data + 1,
					temp4_data+col_k_nnz,
					temp4_rows + 1,
					thrust::greater<HYPRE_Real>());

	     hypre_Memset(temp4_data + lfil + 1, 0, col_k_nnz - lfil - 1, HYPRE_MEMORY_HOST);
	     hypre_Memset(temp4_rows + lfil + 1, n+1, col_k_nnz - lfil - 1, HYPRE_MEMORY_HOST);
	     thrust::stable_sort_by_key(thrust::host, temp4_rows, temp4_rows+lfil, temp4_data);
	     col_k_nnz = lfil + 1;

	     if (temp4_rows[0] != k)
	       {
		 hypre_printf("%s %s %d : Value above the diagonal -- col=%d row=%d value=%g\n",__FILE__,__FUNCTION__,__LINE__, k, temp4_rows[0], temp4_data[0]);
	       }

           }
       }

       Lcsc_nnz += col_k_nnz;
#ifdef HYPRE_ILDL_DEBUG
       hypre_printf("%s %s %d : col_k_nnz=%d\n",__FILE__,__FUNCTION__,__LINE__,Lcsc_nnz);
       fflush(NULL);
#endif

       if (Lcsc_nnz>capacity) {
           /* reallocate */
           Lcsc_rows = hypre_TReAlloc_v2(Lcsc_rows, HYPRE_Int, capacity, HYPRE_Int, capacity+nnz_A, HYPRE_MEMORY_HOST);
           Lcsc_data = hypre_TReAlloc_v2(Lcsc_data, HYPRE_Real, capacity, HYPRE_Real, capacity+nnz_A, HYPRE_MEMORY_HOST);
           capacity = capacity + nnz_A;
           //#ifdef HYPRE_ILDL_DEBUG
           hypre_printf("%s %s %d : k=%d, capacity : before=%d, after=%d\n",__FILE__,__FUNCTION__,__LINE__,k,capacity-nnz_A,capacity);
           fflush(NULL);
           //#endif
       }

       /* exclusive scan */
       Lcsc_col_count[k] = col_k_nnz;
       Lcsc_col_offsets[k]=0;
       for (i=1; i<n+1; ++i) {
           Lcsc_col_offsets[i] = Lcsc_col_count[i-1]+Lcsc_col_offsets[i-1];
       }
       /* next, copy the data onto the end */
       for (i=0; i<col_k_nnz; ++i) {
           if (!std::isfinite(temp4_data[i])) {
	     hypre_printf("%s %s %d : Found infinite value!\n",__FILE__,__FUNCTION__,__LINE__);
               exit(1);
           }
           Lcsc_rows[Lcsc_col_offsets[k]+i] = temp4_rows[i];
           Lcsc_data[Lcsc_col_offsets[k]+i] = temp4_data[i];
       }

       hypre_TFree(temp4_data, HYPRE_MEMORY_HOST);
       hypre_TFree(temp4_rows, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_CUDA
       if (k==lastk+1000 || k==n-1) {
           cudaEventRecord(stop, 0);
           cudaEventSynchronize(stop);
           cudaEventElapsedTime(&time, start, stop);
           printf("%s %s %d : time[%d, %d] = %1.5g\n",__FILE__,__FUNCTION__,__LINE__,lastk,k,time/1000.);
           cudaEventRecord(start, 0);
           lastk=k;
       }
#endif
   }

   hypre_TFree(temp1, HYPRE_MEMORY_HOST);
   hypre_TFree(temp2, HYPRE_MEMORY_HOST);
   hypre_TFree(avect, HYPRE_MEMORY_HOST);
   hypre_TFree(temp3, HYPRE_MEMORY_HOST);

   /* Convert L to CSR */
   HYPRE_Int nnz_L = Lcsc_col_offsets[n];
   hypre_CSRMatrix * Lcsc = hypre_CSRMatrixCreate(n, m, nnz_L);
   hypre_CSRMatrix * Lcsr = hypre_CSRMatrixCreate(n, m, nnz_L);

   hypre_CSRMatrixInitialize_v2( Lcsc, 0, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixInitialize_v2( Lcsr, 0, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixOwnsData(Lcsc) = 0;

#ifdef HYPRE_ILDL_DEBUG
   hypre_printf("%s %s %d : nnz_L=%d\n",__FILE__,__FUNCTION__,__LINE__,nnz_L);
   fflush(NULL);
#endif

   /* Move the factorized data to the matrix data structure */
   hypre_TMemcpy(hypre_CSRMatrixI(Lcsc), Lcsc_col_offsets, HYPRE_Int, n+1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixJ(Lcsc) = Lcsc_rows;
   hypre_CSRMatrixData(Lcsc) = Lcsc_data;

#ifdef HYPRE_ILDL_DEBUG
   hypre_printf("%s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
   fflush(NULL);
#endif

   /* transpose */
   hypre_CSRMatrixTranspose(Lcsc, &Lcsr, 1);

   /* transfer Lcsr back to the Host and construct the full matrix */
   HYPRE_Int * Lcsr_row_offsets = hypre_CSRMatrixI(Lcsr);
   HYPRE_Int * Lcsr_cols = hypre_CSRMatrixJ(Lcsr);
   HYPRE_Real * Lcsr_data = hypre_CSRMatrixData(Lcsr);

   /* attempt to write the factored L/D to a file */
   if (strcmp(mmfilename,"")!=0) {
      hypre_printf("%s %s %d : mmfilename=%s\n",__FILE__,__FUNCTION__,__LINE__,mmfilename);
      FILE * fout = fopen(mmfilename,"wt");
      int ret_code = fprintf(fout, "%%%%MatrixMarket matrix coordinate real general\n%\n");
      fprintf(fout, "%d %d %d\n", n, n, nnz_L);
      for (int i=0; i<n; ++i)
      {
	  for (int j=Lcsc_col_offsets[i]; j<Lcsc_col_offsets[i+1]; ++j)
	  {
	      if (i==Lcsc_rows[j])
		  fprintf(fout,"%d %d %1.15e\n",Lcsc_rows[j]+1,i+1,D_data[i]);
	      else
		  fprintf(fout,"%d %d %1.15e\n",Lcsc_rows[j]+1,i+1,Lcsc_data[j]);
	  }
      }
      fclose(fout);
   }

#ifdef HYPRE_ILDL_DEBUG
   hypre_printf("%s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
   fflush(NULL);
#endif

   /* Create this matrix */
   HYPRE_Int nnz_LDL = n + 2*(nnz_L-n);

   HYPRE_Int * LDL_diag_i = hypre_TAlloc(HYPRE_Int, n + 1, HYPRE_MEMORY_HOST);
   HYPRE_Int * LDL_diag_j = hypre_TAlloc(HYPRE_Int, nnz_LDL, HYPRE_MEMORY_HOST);
   HYPRE_Real * LDL_diag_data = hypre_TAlloc(HYPRE_Real, nnz_LDL, HYPRE_MEMORY_HOST);

   //#ifdef HYPRE_ILDL_DEBUG
   hypre_printf("%s %s %d : n=%d, nnz_L=%d, nnz_LDL=%d\n",__FILE__,__FUNCTION__,__LINE__,n,nnz_L,nnz_LDL);
   fflush(NULL);
   //#endif

   HYPRE_Int pos = 0;
   for (i = 0; i < n; i++)
   {
      LDL_diag_i[i] = pos;
      for (j = Lcsr_row_offsets[i]; j < Lcsr_row_offsets[i+1]-1; j++)
      {
         LDL_diag_j[pos] = Lcsr_cols[j];
         LDL_diag_data[pos++] = Lcsr_data[j];
      }
      LDL_diag_j[pos] = i;
      LDL_diag_data[pos++] = D_data[i];
      for (j = Lcsc_col_offsets[i]+1; j < Lcsc_col_offsets[i+1]; j++)
      {
         LDL_diag_j[pos] = Lcsc_rows[j];
         LDL_diag_data[pos++] = D_data[i] * Lcsc_data[j];
      }
   }
   LDL_diag_i[n] = pos;

   hypre_assert(pos == nnz_LDL);

   /**** BEGIN ADDED by JM ****/
   /* attempt to write the factored L/D to a file */
   if (strcmp(mmfilename,"")!=0) {

     char *buf = NULL;
     asprintf(&buf, "LDL_LU_FORM_%s", mmfilename);

     if (buf != NULL)
     {
       hypre_printf("%s %s %d : mmfilename=%s\n",__FILE__,__FUNCTION__,__LINE__,mmfilename);
       FILE * fout = fopen(buf,"wt");
       int ret_code = fprintf(fout, "%%%%MatrixMarket matrix coordinate real general\n%\n");
       fprintf(fout, "%d %d %d\n", n, n, nnz_LDL);
       for (int i=0; i<n; ++i)
       {
	 for (int j=LDL_diag_i[i]; j<LDL_diag_i[i+1]; ++j)
	 {
	   fprintf(fout,"%d %d %1.15e\n", i, LDL_diag_j[j], LDL_diag_data[j]);
	 }
       }
       fclose(fout);
     }
     else
     {
       hypre_printf("ERROR while dumping LU form of LDL matrix!!!\n");
     }
   }
   /**** END ADDED by JM ****/

#ifdef HYPRE_ILDL_DEBUG
   hypre_printf("%s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
   fflush(NULL);
#endif

   HYPRE_Int * d_LDL_diag_i = hypre_TAlloc(HYPRE_Int, n + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int * d_LDL_diag_j = hypre_TAlloc(HYPRE_Int, nnz_LDL, HYPRE_MEMORY_DEVICE);
   HYPRE_Real * d_LDL_diag_data = hypre_TAlloc(HYPRE_Real, nnz_LDL, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(d_LDL_diag_i, LDL_diag_i, HYPRE_Int, n+1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(d_LDL_diag_j, LDL_diag_j, HYPRE_Int, nnz_LDL, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(d_LDL_diag_data, LDL_diag_data, HYPRE_Real, nnz_LDL, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   hypre_ParCSRMatrix   *LDL = hypre_ParCSRMatrixCreate(comm, n, n, 0, 0, 0, nnz_LDL, 0);
   hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(LDL))    = d_LDL_diag_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(LDL))    = d_LDL_diag_j;
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(LDL)) = d_LDL_diag_data;

   /* Set this pointer */
   *LDLptr = LDL;
#ifdef HYPRE_ILDL_DEBUG
   hypre_printf("%s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
   fflush(NULL);
#endif

   /* destroy these */
   hypre_CSRMatrixDestroy(Lcsc);
   hypre_CSRMatrixDestroy(Lcsr);

   /* Free stuff */
   hypre_TFree(Acsc_col_offsets, HYPRE_MEMORY_HOST);
   hypre_TFree(Acsc_rows, HYPRE_MEMORY_HOST);
   hypre_TFree(Acsc_data, HYPRE_MEMORY_HOST);
   hypre_TFree(Lcsc_col_offsets, HYPRE_MEMORY_HOST);
   hypre_TFree(Lcsc_rows, HYPRE_MEMORY_HOST);
   hypre_TFree(Lcsc_data, HYPRE_MEMORY_HOST);
   hypre_TFree(Lcsc_col_count, HYPRE_MEMORY_HOST);
   hypre_TFree(D_data, HYPRE_MEMORY_HOST);
   hypre_TFree(LDL_diag_i, HYPRE_MEMORY_HOST);
   hypre_TFree(LDL_diag_j, HYPRE_MEMORY_HOST);
   hypre_TFree(LDL_diag_data, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_CUDA
   cudaEventDestroy( start );
   cudaEventDestroy( stop );
#endif

#ifdef HYPRE_ILDL_DEBUG
   hypre_printf("%s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
   fflush(NULL);
#endif

   return hypre_error_flag;
}
