#!/bin/bash
#module load DefApps
module load gcc/10.1.0
module load cuda/11.2
module load helics/helics-3.1.0_openmpi 
#module load spectrum-mpi/10.4.0.3-20210112

#make distclean

export CFLAGS="-g -O2"
export CXXFLAGS="-g -O2"
export HYPRE_CUDA_SM=70
export CXX=$(which g++)
export CC=$(which gcc)
export FC=$(which gfortran)
export F77=$(which gfortran)
export MPICXX=$(which mpicxx)
export MPICC=$(which mpicc)
export MPIFC=$(which mpif90)
export CXX=${MPICXX}
export CC=${MPICC}
export CUDACXX=$(which nvcc)

HYPRE_INSTALL_PREFIX=/home/vbharad2/hypre
MPI_LIB_DIR=/nopt/nrel/apps/openmpi/4.1.0-gcc-8.4.0-j15/lib
MPI_INC_DIR=/nopt/nrel/apps/openmpi/4.1.0-gcc-8.4.0-j15/include

./configure --prefix=$HYPRE_INSTALL_PREFIX --with-MPI --with-MPI-lib-dirs=$MPI_LIB_DIR --with-MPI-include=$MPI_INC_DIR --without-openmp --disable-bigint --disable-mixedint --disable-complex --enable-shared --without-superlu --without-mli --disable-debug --with-cuda --enable-curand --enable-cusparse --with-gpu-arch=70 --disable-fortran #--enable-unified-memory --enable-device-memory-pool

make -j8
make install

