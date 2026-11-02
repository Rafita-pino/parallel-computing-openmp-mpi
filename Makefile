CC      ?= gcc
MPICC   ?= mpicc
CFLAGS  ?= -O3

OMP_BINS = 01-kmeans-openmp/kmeans_omp 02-sparse-matmul-openmp/spmm_omp
MPI_BINS = 03-kmeans-mpi/kmeans_mpi 04-sparse-matmul-mpi/spmm_mpi

.PHONY: all omp mpi clean

all: omp mpi

omp: $(OMP_BINS)

mpi: $(MPI_BINS)

01-kmeans-openmp/kmeans_omp: 01-kmeans-openmp/kmeans_omp.c
	$(CC) $(CFLAGS) -fopenmp $< -o $@

02-sparse-matmul-openmp/spmm_omp: 02-sparse-matmul-openmp/spmm_omp.c
	$(CC) $(CFLAGS) -fopenmp $< -o $@

03-kmeans-mpi/kmeans_mpi: 03-kmeans-mpi/kmeans_mpi.c
	$(MPICC) $(CFLAGS) $< -o $@

04-sparse-matmul-mpi/spmm_mpi: 04-sparse-matmul-mpi/spmm_mpi.c
	$(MPICC) $(CFLAGS) $< -o $@

clean:
	rm -f $(OMP_BINS) $(MPI_BINS)
