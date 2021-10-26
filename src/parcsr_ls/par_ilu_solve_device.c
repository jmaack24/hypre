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

/*********************************************************************************/
/*                   hypre_ILUSolveDeviceLU                                      */
/*********************************************************************************/
/* Incomplete LU solve (GPU)
 * L, D and U factors only have local scope (no off-diagonal processor terms)
 * so apart from the residual calculation (which uses A), the solves with the
 * L and U factors are local.
*/

HYPRE_Int
hypre_ILUSolveDeviceLU(hypre_ParCSRMatrix *A, hypre_GpuMatData * matL_des,
                       hypre_GpuMatData * matU_des, hypre_CsrsvData * matLU_csrsvdata,
                       hypre_CSRMatrix *matLU_d, hypre_ParVector *f,  hypre_ParVector *u, HYPRE_Int *perm,
                       HYPRE_Int n, hypre_ParVector *ftemp, hypre_ParVector *utemp)
{
#if defined(HYPRE_USING_CUSPARSE)
   hypre_ILUSolveCusparseLU(A, matL_des, matU_des, matLU_csrsvdata,
                            matLU_d, f,  u, perm, n, ftemp, utemp);
#endif

#if defined(HYPRE_USING_ROCSPARSE)
   hypre_ILUSolveRocsparseLU(A, matL_des, matU_des, matLU_csrsvdata,
                             matLU_d, f,  u, perm, n, ftemp, utemp);
#endif
   return hypre_error_flag;
}

#if defined(HYPRE_USING_CUSPARSE)

HYPRE_Int
hypre_ILUSolveCusparseLU(hypre_ParCSRMatrix *A, hypre_GpuMatData * matL_des,
                         hypre_GpuMatData * matU_des, hypre_CsrsvData * matLU_csrsvdata,
                         hypre_CSRMatrix *matLU_d,
                         hypre_ParVector *f,  hypre_ParVector *u, HYPRE_Int *perm,
                         HYPRE_Int n, hypre_ParVector *ftemp, hypre_ParVector *utemp)
{
   /* Only solve when we have stuffs to be solved */
   if (n == 0)
   {
      return hypre_error_flag;
   }

   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int my_id;
   hypre_MPI_Comm_rank(comm, &my_id);

   /* ILU data */
   HYPRE_Real              *LU_data             = hypre_CSRMatrixData(matLU_d);
   HYPRE_Int               *LU_i                = hypre_CSRMatrixI(matLU_d);
   HYPRE_Int               *LU_j                = hypre_CSRMatrixJ(matLU_d);
   HYPRE_Int               nnz                  = hypre_CSRMatrixNumNonzeros(matLU_d);

   hypre_Vector            *utemp_local         = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real              *utemp_data          = hypre_VectorData(utemp_local);

   hypre_Vector            *ftemp_local         = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real              *ftemp_data          = hypre_VectorData(ftemp_local);

   HYPRE_Real              alpha;
   HYPRE_Real              beta;
   //HYPRE_Int               i, j, k1, k2;

   HYPRE_Int               isDoublePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int               isSinglePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   hypre_assert(isDoublePrecision || isSinglePrecision);

   /* begin */
   alpha = -1.0;
   beta = 1.0;

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());

   /* Initialize Utemp to zero.
    * This is necessary for correctness, when we use optimized
    * vector operations in the case where sizeof(L, D or U) < sizeof(A)
   */
   //hypre_ParVectorSetConstantValues( utemp, 0.);
   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* apply permutation */
   HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);

   if (isDoublePrecision)
   {
      /* L solve - Forward solve */
      HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                n, nnz, (hypre_double *) &beta, hypre_GpuMatDataMatDescr(matL_des),
                                                (hypre_double *) LU_data, LU_i, LU_j, hypre_CsrsvDataInfoL(matLU_csrsvdata),
                                                (hypre_double *) utemp_data, (hypre_double *) ftemp_data,
                                                hypre_CsrsvDataSolvePolicy(matLU_csrsvdata), hypre_CsrsvDataBuffer(matLU_csrsvdata) ));


#ifdef WRITE_TO_FILE_DEBUG
      cudaDeviceSynchronize();
      char name[50];
      sprintf(name,"LDirect_%d.txt",my_id);
      writeToFileDebug(ftemp_local, n, 0, name);
#endif

      /* U solve - Backward substitution */
      HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                n, nnz, (hypre_double *) &beta, hypre_GpuMatDataMatDescr(matU_des),
                                                (hypre_double *) LU_data, LU_i, LU_j, hypre_CsrsvDataInfoU(matLU_csrsvdata),
                                                (hypre_double *) ftemp_data, (hypre_double *) utemp_data,
                                                hypre_CsrsvDataSolvePolicy(matLU_csrsvdata), hypre_CsrsvDataBuffer(matLU_csrsvdata) ));
#ifdef WRITE_TO_FILE_DEBUG
      cudaDeviceSynchronize();
      sprintf(name,"UDirect_%d.txt",my_id);
      writeToFileDebug(utemp_local, n, 0, name);
#endif
   }
   else if (isSinglePrecision)
   {
      /* L solve - Forward solve */
      HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                n, nnz, (float *) &beta, hypre_GpuMatDataMatDescr(matL_des),
                                                (float *) LU_data, LU_i, LU_j, hypre_CsrsvDataInfoL(matLU_csrsvdata),
                                                (float *) utemp_data, (float *) ftemp_data,
                                                hypre_CsrsvDataSolvePolicy(matLU_csrsvdata), hypre_CsrsvDataBuffer(matLU_csrsvdata) ));

      /* U solve - Backward substitution */
      HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                n, nnz, (float *) &beta, hypre_GpuMatDataMatDescr(matU_des),
                                                (float *) LU_data, LU_i, LU_j, hypre_CsrsvDataInfoU(matLU_csrsvdata),
                                                (float *) ftemp_data, (float *) utemp_data,
                                                hypre_CsrsvDataSolvePolicy(matLU_csrsvdata), hypre_CsrsvDataBuffer(matLU_csrsvdata) ));
   }

   /* apply reverse permutation */
   HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + n, perm, ftemp_data);
   /* Update solution */
   hypre_ParVectorAxpy(beta, ftemp, u);


   return hypre_error_flag;
}

#endif

#if defined(HYPRE_USING_ROCSPARSE)

HYPRE_Int
hypre_ILUSolveRocsparseLU(hypre_ParCSRMatrix *A, hypre_GpuMatData * matL_des,
                          hypre_GpuMatData * matU_des, hypre_CsrsvData * matLU_csrsvdata,
                          hypre_CSRMatrix *matLU_d,
                          hypre_ParVector *f,  hypre_ParVector *u, HYPRE_Int *perm,
                          HYPRE_Int n, hypre_ParVector *ftemp, hypre_ParVector *utemp)
{
   /* Only solve when we have stuffs to be solved */
   if (n == 0)
   {
      return hypre_error_flag;
   }

   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int my_id;
   hypre_MPI_Comm_rank(comm, &my_id);

   /* ILU data */
   HYPRE_Real              *LU_data             = hypre_CSRMatrixData(matLU_d);
   HYPRE_Int               *LU_i                = hypre_CSRMatrixI(matLU_d);
   HYPRE_Int               *LU_j                = hypre_CSRMatrixJ(matLU_d);
   HYPRE_Int               nnz                  = hypre_CSRMatrixNumNonzeros(matLU_d);

   hypre_Vector            *utemp_local         = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real              *utemp_data          = hypre_VectorData(utemp_local);

   hypre_Vector            *ftemp_local         = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real              *ftemp_data          = hypre_VectorData(ftemp_local);

   HYPRE_Real              alpha;
   HYPRE_Real              beta;
   //HYPRE_Int               i, j, k1, k2;

   HYPRE_Int               isDoublePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int               isSinglePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   hypre_assert(isDoublePrecision || isSinglePrecision);

   /* begin */
   alpha = -1.0;
   beta = 1.0;

   rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());

   /* Initialize Utemp to zero.
    * This is necessary for correctness, when we use optimized
    * vector operations in the case where sizeof(L, D or U) < sizeof(A)
   */
   //hypre_ParVectorSetConstantValues( utemp, 0.);
   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* apply permutation */
   HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);

   if (isDoublePrecision)
   {
      /* L solve - Forward solve */
      HYPRE_ROCSPARSE_CALL( rocsparse_dcsrsv_solve(handle, rocsparse_operation_none, n, nnz,
                                                   (hypre_double *) &beta, hypre_GpuMatDataMatDescr(matL_des),
                                                   (hypre_double *) LU_data, LU_i, LU_j, hypre_CsrsvDataInfoL(matLU_csrsvdata),
                                                   (hypre_double *) utemp_data, (hypre_double *) ftemp_data,
                                                   hypre_CsrsvDataSolvePolicy(matLU_csrsvdata), hypre_CsrsvDataBuffer(matLU_csrsvdata) ));

      /* U solve - Backward substitution */
      HYPRE_ROCSPARSE_CALL( rocsparse_dcsrsv_solve(handle, rocsparse_operation_none, n, nnz,
                                                   (hypre_double *) &beta, hypre_GpuMatDataMatDescr(matU_des),
                                                   (hypre_double *) LU_data, LU_i, LU_j, hypre_CsrsvDataInfoU(matLU_csrsvdata),
                                                   (hypre_double *) ftemp_data, (hypre_double *) utemp_data,
                                                   hypre_CsrsvDataSolvePolicy(matLU_csrsvdata), hypre_CsrsvDataBuffer(matLU_csrsvdata) ));
   }
   else if (isSinglePrecision)
   {
      /* L solve - Forward solve */
      HYPRE_ROCSPARSE_CALL( rocsparse_scsrsv_solve(handle, rocsparse_operation_none, n, nnz,
                                                   (float *) &beta, hypre_GpuMatDataMatDescr(matL_des),
                                                   (float *) LU_data, LU_i, LU_j, hypre_CsrsvDataInfoL(matLU_csrsvdata),
                                                   (float *) utemp_data, (float *) ftemp_data,
                                                   hypre_CsrsvDataSolvePolicy(matLU_csrsvdata), hypre_CsrsvDataBuffer(matLU_csrsvdata) ));

      /* U solve - Backward substitution */
      HYPRE_ROCSPARSE_CALL( rocsparse_scsrsv_solve(handle, rocsparse_operation_none, n, nnz,
                                                   (float *) &beta, hypre_GpuMatDataMatDescr(matU_des),
                                                   (float *) LU_data, LU_i, LU_j, hypre_CsrsvDataInfoU(matLU_csrsvdata),
                                                   (float *) ftemp_data, (float *) utemp_data,
                                                   hypre_CsrsvDataSolvePolicy(matLU_csrsvdata), hypre_CsrsvDataBuffer(matLU_csrsvdata) ));
   }

   /* apply reverse permutation */
   HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + n, perm, ftemp_data);
   /* Update solution */
   hypre_ParVectorAxpy(beta, ftemp, u);


   return hypre_error_flag;
}

#endif
