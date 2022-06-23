#!/bin/bash
# module load gcc/10.1.0
# module load cuda/11.2
# module load helics/helics-3.1.0_openmpi 


#make distclean

export CFLAGS="-g -O2"
export CXXFLAGS="-g -O2"
export HYPRE_CUDA_SM=70
export CXX=$(which g++)
export CC=$(which gcc)
export FC=$(which gfortran)
export F77=$(which gfortran)
# export MPICXX=$(which mpicxx)
# export MPICC=$(which mpicc)
# export MPIFC=$(which mpif90)
# export CXX=${MPICXX}
# export CC=${MPICC}
export CUDACXX=$(which nvcc)

# MPI_LIB_DIR=${OPENMPI_ROOT_DIR}/lib
# MPI_INC_DIR=${OPENMPI_ROOT_DIR}/include

echo $CUDA_HOME
# ./configure --prefix=${MYAPPS} --with-MPI --with-MPI-lib-dirs=$MPI_LIB_DIR --with-MPI-include=$MPI_INC_DIR --without-openmp --disable-bigint --disable-mixedint --disable-complex --enable-shared --without-superlu --without-mli --disable-debug --with-cuda --enable-curand --enable-cusparse --with-gpu-arch=70 --disable-fortran #--enable-unified-memory --enable-device-memory-pool

./configure --prefix=${MYAPPS} --without-MPI --without-openmp --disable-bigint --disable-mixedint --disable-complex --enable-shared --without-superlu --without-mli --enable-debug --with-cuda --enable-curand --enable-cusparse --with-gpu-arch=70 --disable-fortran #--enable-unified-memory --enable-device-memory-pool

make -j8 && make install

#make install
