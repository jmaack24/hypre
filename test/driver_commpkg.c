/* driver_commpkg.c*/
/* AHB 06/04 */
/* purpose:  to test a new communication package for the ij interface */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#include "utilities.h"
#include "parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"

/* #include "parcsr_ls.h"
 #include "HYPRE.h" 
 #include "HYPRE_parcsr_mv.h" 
 #include "krylov.h"  */



/*some debugging tools*/
#define   mydebug 0
#define   mpip_on 0

/*time an allgather in addition to the current commpkg */
#define   time_gather 1

/* for timing multiple commpkg setup (if you want the time to be larger in the
   hopes of getting smaller stds - often not effective) */
#define   LOOP2  1


int myBuildParLaplacian (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int myBuildParLaplacian27pt (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );


void stats_mo(double*, int, double *,double *);

/*==========================================================================*/


/*------------------------------------------------------------------
 *
 * This tests an alternate comm package for ij
 *
 * options:
 *         -laplacian              3D 7pt stencil
 *         -27pt                   3D 27pt laplacian
 *         -fromonecsrfile         read matrix from a csr file
 *         -commpkg <int>          1 = new comm. package
 *                                 2  =old
 *                                 3 = both (default)
 *         -loop <int>             number of times to loop
 *         -verbose                print more error checking   
 *-------------------------------------------------------------------*/


int
main( int   argc,
      char *argv[] )
{


   int        num_procs, myid;
   int        verbose = 0, build_matrix_type = 1;
   int        index, matrix_arg_index, commpkg_flag=3;
   int        i, k, ierr=0;
   int        row_start, row_end; 
   int        col_start, col_end, global_num_rows;
   int       *row_part, *col_part; 
   char      *csrfilename;
   int        preload = 0, loop = 1, loop2 = LOOP2;   
   int        bcast_rows[2], *info;
   


   hypre_ParCSRMatrix    *parcsr_A, *small_A;
   HYPRE_ParCSRMatrix    A_temp, A_temp_small; 
   hypre_CSRMatrix       *A_CSR;
   hypre_ParCSRCommPkg	 *comm_pkg;   

  
   int                 nx, ny, nz;
   int                 P, Q, R;
   int                 p, q, r;
   double              values[4];

   hypre_ParVector     *x_new, *x;
   hypre_ParVector     *y_new, *y;
   int                 *row_starts;
   double              ans;
   double              start_time, end_time, total_time, *loop_times;
   double              T_avg, T_std;
   

#if mydebug   
   int  j, tmp_int;
#endif

   /*-----------------------------------------------------------
    * Initialize MPI
    *-----------------------------------------------------------*/


   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );


   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
 
   index = 1;
   while ( index < argc) 
   {
      if  ( strcmp(argv[index], "-verbose") == 0 )
      {
         index++;  
         verbose = 1;
      }
      else if ( strcmp(argv[index], "-fromonecsrfile") == 0 )
      {
         index++;
         build_matrix_type      = 1;      
         matrix_arg_index = index; /*this tells where the name is*/
      }
      else if  ( strcmp(argv[index], "-commpkg") == 0 )
      {
         index++;  
         commpkg_flag = atoi(argv[index++]);
      }
      else if ( strcmp(argv[index], "-laplacian") == 0 )
      {
         index++;
         build_matrix_type      = 2;
         matrix_arg_index = index;
      }
      else if ( strcmp(argv[index], "-27pt") == 0 )
      {
         index++;
         build_matrix_type      = 4;
         matrix_arg_index = index;
      }
/*
      else if  ( strcmp(argv[index], "-nopreload") == 0 )
      {
         index++;  
         preload = 0;
      }
*/
      else if  ( strcmp(argv[index], "-loop") == 0 )
      {
         index++;  
         loop = atoi(argv[index++]);
      }

      else  
      {
	 index++;
         /*printf("Warning: Unrecogized option '%s'\n",argv[index++] );*/
      }
   }

   
  
   /*-----------------------------------------------------------
    * Setup the Matrix problem   
    *-----------------------------------------------------------*/

  /*-----------------------------------------------------------
    *  Get actual partitioning- 
    *  read in an actual csr matrix.
    *-----------------------------------------------------------*/


   if (build_matrix_type ==1) /*read in a csr matrix from one file */
   {
      if (matrix_arg_index < argc)
      {
	 csrfilename = argv[matrix_arg_index];
      }
      else
      {
         printf("Error: No filename specified \n");
         exit(1);
      }
      if (myid == 0)
      {
	/*printf("  FromFile: %s\n", csrfilename);*/
         A_CSR = hypre_CSRMatrixRead(csrfilename);
      }
      row_part = NULL;
      col_part = NULL;

      parcsr_A = hypre_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, A_CSR, 
					       row_part, col_part);

      if (myid == 0) hypre_CSRMatrixDestroy(A_CSR);
   }
   else if (build_matrix_type ==2)
   {
      
     myBuildParLaplacian(argc, argv, matrix_arg_index,  &A_temp);
     parcsr_A = (hypre_ParCSRMatrix *) A_temp;      
 
   }
   else if (build_matrix_type ==4)
   {
     myBuildParLaplacian27pt(argc, argv, matrix_arg_index, &A_temp);
     parcsr_A = (hypre_ParCSRMatrix *) A_temp;
   }

 
  /*-----------------------------------------------------------
   * create a small problem so that timings are more accurate - 
   * code gets run twice (small laplace)
   *-----------------------------------------------------------*/

   /*this is no longer being used - preload = 0 is set at the beginning */

   if (preload == 1) 
   {
 
      /*printf("preload!\n");*/
      
        
       values[1] = -1;
       values[2] = -1;
       values[3] = -1;
       values[0] = - 6.0    ;

       nx = 2;
       ny = num_procs;
       nz = 2;

       P  = 1;
       Q  = num_procs;
       R  = 1;

       p = myid % P;
       q = (( myid - p)/P) % Q;
       r = ( myid - p - P*q)/( P*Q );
       
      A_temp_small = (HYPRE_ParCSRMatrix) GenerateLaplacian(MPI_COMM_WORLD, nx, ny, nz, 
				      P, Q, R, p, q, r, values);
      small_A = (hypre_ParCSRMatrix *) A_temp_small;     

      /*do comm packages*/
      hypre_NewCommPkgCreate(small_A);
      hypre_NewCommPkgDestroy(small_A); 

      hypre_MatvecCommPkgCreate(small_A);
      hypre_ParCSRMatrixDestroy(small_A); 
  
   }


   /* instead of preloading, let's not time the first one*/

   loop += 1;
   



   /*-----------------------------------------------------------
    *  Prepare for timing
    *-----------------------------------------------------------*/


   loop_times = hypre_CTAlloc(double, loop);
   


/******************************************************************************************/   

   MPI_Barrier(MPI_COMM_WORLD);

   if (commpkg_flag == 1 || commpkg_flag ==3 )
   {
  
      /*-----------------------------------------------------------
       *  Create new comm package
       *-----------------------------------------------------------*/


    
      if (!myid) printf("********************************************************\n" );  
 
      /*do loop times*/
      for (i=0; i< loop; i++) 
      {
         loop_times[i] = 0.0;
         for (k=0; k< loop2; k++) 
         {
         
            MPI_Barrier(MPI_COMM_WORLD);

            start_time = MPI_Wtime();

#if mpip_on
             if (i==(loop-1)) MPI_Pcontrol(1); 
#endif
     
            hypre_NewCommPkgCreate(parcsr_A);

#if mpip_on
             if (i==(loop-1)) MPI_Pcontrol(0); 
#endif  
  
            end_time = MPI_Wtime();
      
            end_time = end_time - start_time;
        
            MPI_Allreduce(&end_time, &total_time, 1,
                       MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         
            loop_times[i] += total_time;

            if (  !((i+1)== loop  &&  (k+1) == loop2)) hypre_NewCommPkgDestroy(parcsr_A); 
            
      }
      
         if (!myid && i ) printf("....loop[%d] (%d total) =  %f seconds\n", i, loop2, loop_times[i]);  

      }
      

      /* calculate the avg and std. */
      stats_mo(loop_times, loop, &T_avg, &T_std);
    
      if (!myid) printf(" NewCommPkgCreate:  AVG. wall clock time =  %f seconds\n", T_avg);  
      if (!myid) printf("                    STD. for %d  runs     =  %f\n", loop-1, T_std);  
      if (!myid) printf("********************************************************\n" );  
      /*-----------------------------------------------------------
       *  Verbose printing
       *-----------------------------------------------------------*/

      /*some verification*/

       global_num_rows = hypre_ParCSRMatrixGlobalNumRows(parcsr_A); 

       if (verbose) 
       {

	  ierr = hypre_ParCSRMatrixGetLocalRange( parcsr_A,
                                      &row_start, &row_end ,
                                       &col_start, &col_end );


	  comm_pkg = hypre_ParCSRMatrixCommPkg(parcsr_A);
     
          printf("myid = %i, my ACTUAL local range: [%i, %i]\n", myid, 
		 row_start, row_end);
	  
	
	  ierr = hypre_GetAssumedPartitionRowRange( myid, global_num_rows, &row_start, 
					      &row_end);


	  printf("myid = %i, my assumed local range: [%i, %i]\n", myid, 
		 row_start, row_end);

          printf("myid = %d, num_recvs = %d\n", myid, 
		 hypre_ParCSRCommPkgNumRecvs(comm_pkg)  );  

#if mydebug   
	  for (i=0; i < hypre_ParCSRCommPkgNumRecvs(comm_pkg); i++) 
	  {
              printf("myid = %d, recv proc = %d, vec_starts = [%d : %d]\n", 
		     myid,  hypre_ParCSRCommPkgRecvProcs(comm_pkg)[i], 
		     hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)[i],
		     hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)[i+1]-1);
	   }
#endif 
	  printf("myid = %d, num_sends = %d\n", myid, 
		 hypre_ParCSRCommPkgNumSends(comm_pkg)  );  

#if mydebug
	  for (i=0; i <hypre_ParCSRCommPkgNumSends(comm_pkg) ; i++) 
          {
	    tmp_int =  hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i+1] -  
                     hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i];
	    index = hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i];
	    for (j=0; j< tmp_int; j++) 
	    {
	       printf("myid = %d, send proc = %d, send element = %d\n",myid,  
		      hypre_ParCSRCommPkgSendProcs(comm_pkg)[i],
		      hypre_ParCSRCommPkgSendMapElmts(comm_pkg)[index+j]); 
	     }   
	  }
#endif
       }
       /*-----------------------------------------------------------
        *  To verify correctness (if commpkg_flag = 3)
        *-----------------------------------------------------------*/

       if (commpkg_flag == 3 ) 
       {
          /*do a matvec - we are assuming a square matrix */
	 row_starts = hypre_ParCSRMatrixRowStarts(parcsr_A);
   
	 x_new = hypre_ParVectorCreate(MPI_COMM_WORLD, global_num_rows, row_starts);
	 hypre_ParVectorSetPartitioningOwner(x_new, 0);
	 hypre_ParVectorInitialize(x_new);
	 hypre_ParVectorSetConstantValues(x_new, 1.0);    

	 y_new = hypre_ParVectorCreate(MPI_COMM_WORLD, global_num_rows, row_starts);
	 hypre_ParVectorSetPartitioningOwner(y_new, 0);
	 hypre_ParVectorInitialize(y_new);
	 hypre_ParVectorSetConstantValues(y_new, 0.0);

         /*y = 1.0*A*x+1.0*y */
	 hypre_ParCSRMatrixMatvec (1.0, parcsr_A, x_new, 1.0, y_new);
       }
   
   /*-----------------------------------------------------------
    *  Clean up after MyComm
    *-----------------------------------------------------------*/


       hypre_NewCommPkgDestroy(parcsr_A); 

   }

  
/******************************************************************************************/
/******************************************************************************************/

   MPI_Barrier(MPI_COMM_WORLD);


   if (commpkg_flag > 1 )
   {

      /*-----------------------------------------------------------
       *  Set up standard comm package
       *-----------------------------------------------------------*/

      bcast_rows[0] = 23;
      bcast_rows[1] = 1789;
      
      if (!myid) printf("********************************************************\n" );  
      /*do loop times*/
      for (i=0; i< loop; i++) 
      {

         loop_times[i] = 0.0;
         for (k=0; k< loop2; k++) 
         {
            

            MPI_Barrier(MPI_COMM_WORLD);

         
            start_time = MPI_Wtime();

#if time_gather
                  
            info = hypre_CTAlloc(int, num_procs);
            
            MPI_Allgather(bcast_rows, 1, MPI_INT, info, 1, MPI_INT, MPI_COMM_WORLD); 

#endif

            hypre_MatvecCommPkgCreate(parcsr_A);

            end_time = MPI_Wtime();


            end_time = end_time - start_time;
        
            MPI_Allreduce(&end_time, &total_time, 1,
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            loop_times[i] += total_time;
         
       
         if (  !((i+1)== loop  &&  (k+1) == loop2))   hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkg(parcsr_A));
               
        }
         
        
         if (!myid && i) printf("....loop[%d] (%d total)=  %f seconds\n", i, loop2, loop_times[i]);  
       
         
            
      }
      
      stats_mo(loop_times, loop, &T_avg, &T_std);      
      if (!myid) printf("Current CommPkgCreate:  AVG. wall clock time =  %f seconds\n", T_avg);  
      if (!myid) printf("                        STD. for %d  runs     =  %f\n", loop-1, T_std);  
      if (!myid) printf("********************************************************\n" );  

      /*-----------------------------------------------------------
       * Verbose printing
       *-----------------------------------------------------------*/

      /*some verification*/

    
       if (verbose) 
       {

          ierr = hypre_ParCSRMatrixGetLocalRange( parcsr_A,
						  &row_start, &row_end ,
						  &col_start, &col_end );


          comm_pkg = hypre_ParCSRMatrixCommPkg(parcsr_A);
     
          printf("myid = %i, std - my local range: [%i, %i]\n", myid, 
		 row_start, row_end);

          ierr = hypre_ParCSRMatrixGetLocalRange( parcsr_A,
						  &row_start, &row_end ,
						  &col_start, &col_end );

          printf("myid = %d, std - num_recvs = %d\n", myid, 
		 hypre_ParCSRCommPkgNumRecvs(comm_pkg)  );  

#if mydebug   
	  for (i=0; i < hypre_ParCSRCommPkgNumRecvs(comm_pkg); i++) 
          {
              printf("myid = %d, std - recv proc = %d, vec_starts = [%d : %d]\n", 
		     myid,  hypre_ParCSRCommPkgRecvProcs(comm_pkg)[i], 
		     hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)[i],
		     hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)[i+1]-1);
	  }
#endif
          printf("myid = %d, std - num_sends = %d\n", myid, 
		 hypre_ParCSRCommPkgNumSends(comm_pkg));  


#if mydebug
          for (i=0; i <hypre_ParCSRCommPkgNumSends(comm_pkg) ; i++) 
          {
	     tmp_int =  hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i+1] -  
	                hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i];
	     index = hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i];
	     for (j=0; j< tmp_int; j++) 
	     {
	        printf("myid = %d, std - send proc = %d, send element = %d\n",myid,  
		       hypre_ParCSRCommPkgSendProcs(comm_pkg)[i],
		       hypre_ParCSRCommPkgSendMapElmts(comm_pkg)[index+j]); 
	     }   
	  } 
#endif
       }

       /*-----------------------------------------------------------
        * Verify correctness
        *-----------------------------------------------------------*/

 

       if (commpkg_flag == 3 ) 
       { 
          global_num_rows = hypre_ParCSRMatrixGlobalNumRows(parcsr_A); 
          row_starts = hypre_ParCSRMatrixRowStarts(parcsr_A);
 
       
          x = hypre_ParVectorCreate(MPI_COMM_WORLD, global_num_rows, row_starts);
          hypre_ParVectorSetPartitioningOwner(x, 0);
          hypre_ParVectorInitialize(x);
          hypre_ParVectorSetConstantValues(x ,1.0);    

          y = hypre_ParVectorCreate(MPI_COMM_WORLD, global_num_rows,row_starts);
          hypre_ParVectorSetPartitioningOwner(y, 0);
          hypre_ParVectorInitialize(y);
          hypre_ParVectorSetConstantValues(y, 0.0);

          hypre_ParCSRMatrixMatvec (1.0, parcsr_A, x, 1.0, y);
      
       }

   }





   /*-----------------------------------------------------------
    *  Compare matvecs for both comm packages (3)
    *-----------------------------------------------------------*/

   if (commpkg_flag == 3 ) 
   { 
     /*make sure that y and y_new are the same  - now y_new should=0*/   
     hypre_ParVectorAxpy( -1.0, y, y_new );


     hypre_ParVectorSetRandomValues(y, 1);

     ans = hypre_ParVectorInnerProd( y, y_new );
     if (!myid)
     {
        
        if ( fabs(ans) > 1e-8 ) 
        {  
           printf("!!!!! WARNING !!!!! should be zero if correct = %6.10f\n", 
                  ans); 
        } 
        else
        {
           printf("Matvecs match ( should be zero = %6.10f )\n", 
                  ans); 
        }
     }
     

   }
 

   /*-----------------------------------------------------------
    *  Clean up
    *-----------------------------------------------------------*/

    
   hypre_ParCSRMatrixDestroy(parcsr_A); /*this calls the standard comm 
                                          package destroy - but we'll destroy 
                                          ours separately until it is
                                          incorporated */

  if (commpkg_flag == 3 ) 
  { 
      hypre_ParVectorDestroy(x);
      hypre_ParVectorDestroy(x_new);
      hypre_ParVectorDestroy(y);
      hypre_ParVectorDestroy(y_new);
  }




   MPI_Finalize();

   return(ierr);


}





/*------------------------------------
 *    Calculate the average and STD   
 *     throw away 1st timing       
 *------------------------------------*/

void stats_mo(double array[], int n, double *Tavg,double *Tstd)
{

    int i;
    double atmp, tmp=0.0;
    double avg = 0.0, std;

  
    for(i=1; i<n; i++) {
       atmp = array[i];
       avg += atmp;
       tmp += atmp*atmp;
    }

    n = n-1;    
    avg = avg/(double) n;
    tmp = tmp/(double) n;

    tmp = fabs(tmp - avg*avg);
    std = sqrt(tmp);

    *Tavg = avg;
    *Tstd = std;
}



/*These next two functions are from ij.c in linear_solvers/tests */


/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
myBuildParLaplacian27pt( int                  argc,
                       char                *argv[],
                       int                  arg_index,
                       HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian_27pt:\n");
      printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 2);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
	values[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
	values[0] = 2.0;
   values[1] = -1.;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(MPI_COMM_WORLD,
                               nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}


/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/


int
myBuildParLaplacian( int                  argc,
                   char                *argv[],
                   int                  arg_index,
                   HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;
   double              cx, cy, cz;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian:\n");
      printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 4);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(MPI_COMM_WORLD, nx, ny, nz, 
					      P, Q, R, p, q, r, values);

   hypre_TFree(values);


   *A_ptr = A;

   return (0);
}