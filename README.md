# Parallel Computing with OpenMP & MPI

Four high-performance computing projects in C: two classic kernels (**K-means clustering** and **sparse matrix multiplication**) parallelised first for shared memory with **OpenMP** and then for distributed memory with **MPI**.

All of them were benchmarked on the SLURM cluster of the Universitat Rovira i Virgili, from 2 up to 128 threads on a single node and from 2 to 16 nodes (up to 512 MPI processes). Every parallel run is checked automatically against the output of the sequential version.

> Coursework for *Computació Paral·lela i Massiva* (Parallel and Massive Computing), BSc in Computer Engineering, URV, 2025–26.

## Results at a glance

| # | Project | Model | Problem size | Best speedup | Harmonic-mean speedup (target) |
|---|---------|-------|--------------|--------------|--------------------------------|
| 01 | [K-means](01-kmeans-openmp/) | OpenMP | 600 000 points, 200 clusters | **72.1×** (128 threads) | Orca 6.32 (≥ 6) · Teen 3.28 (≥ 4) |
| 02 | [Sparse matrix multiplication](02-sparse-matmul-openmp/) | OpenMP | 8000 × 8000, 1 % non-zeros | **23.4×** (32 threads) | Orca 12.79 (≥ 5) · Teen 4.62 (≥ 3.5) |
| 03 | [K-means](03-kmeans-mpi/) | MPI | 2 000 000 points, 400 clusters | **19.4×** (16 nodes × 8) | 6.27 (≥ 4) |
| 04 | [Sparse matrix multiplication](04-sparse-matmul-mpi/) | MPI | 8000 × 8000, 1 % non-zeros | **24.7×** (16 nodes × 2) | 10.93 (no target) |

Speedups are measured against the sequential times given in the assignments (`gcc -O3`). The harmonic mean is taken over all the required thread or process configurations, so it penalises the configurations that scale worst. Each project's README has the full tables.

## Techniques used

**Shared memory (OpenMP)**
- Data-dependence analysis to find what can actually run in parallel (RAW / WAR / WAW, loop-carried dependences).
- Array-section reductions (`reduction(+:fS[:G])`) so per-thread partial sums stay in L1 cache instead of fighting over shared memory.
- Manual loop unrolling ×8, matched to the 64-byte cache line, to expose instruction-level parallelism.
- `nowait` combined with `schedule(static)` to remove barriers that aren't needed.
- Cache blocking (loop tiling) and unit-stride access for the dense operand.
- "Dirty-index" tracking, so only the touched entries of the work vectors are reset.
- Bulk slot reservation with a single `atomic capture` per column instead of one atomic per element.
- Task parallelism with `sections`, overlapping two independent sorts.

**Distributed memory (MPI)**
- Block domain decomposition with balanced remainders.
- *Shadow generation*: every rank reproduces the same pseudo-random input from a shared seed and keeps only its own slice, so the input needs no `Scatter`.
- `MPI_Allreduce` on a packed buffer (sum and count interleaved): one collective per iteration and no master bottleneck.
- A single packed `MPI_Reduce` for the final statistics.
- 64-byte aligned allocations (`posix_memalign`, `aligned(64)`) and padded structs to avoid cache-line splits.
- Benchmark scripts that allocate all the SLURM nodes once and run every configuration inside that allocation, so the timings don't include queueing time.

## Repository layout

```
01-kmeans-openmp/          K-means, OpenMP
02-sparse-matmul-openmp/   Sparse x dense / sparse x sparse products, OpenMP
03-kmeans-mpi/             K-means, MPI
04-sparse-matmul-mpi/      Sparse matrix products, MPI
Makefile                   Builds everything locally
```

Each folder contains the source code, a `run_benchmarks.sh` script (written for the URV SLURM partitions) and a README explaining the approach and the results.

## Build and run locally

Requirements: GCC with OpenMP support, plus an MPI implementation (Open MPI or MPICH) for projects 03 and 04.

```bash
make            # builds all four projects (make omp / make mpi for one family)

OMP_NUM_THREADS=8 ./01-kmeans-openmp/kmeans_omp
OMP_NUM_THREADS=8 ./02-sparse-matmul-openmp/spmm_omp      # needs ~2.1 GB RAM
mpirun -np 4 ./03-kmeans-mpi/kmeans_mpi
mpirun -np 4 ./04-sparse-matmul-mpi/spmm_mpi 8000          # ~2 GB RAM per process
```

Problem sizes are fixed in the sources (as required by the assignments), except for project 04, which takes `N` as an argument.

## Test systems

| Cluster | Used for | Hardware |
|---------|----------|----------|
| Teen | OpenMP | AMD Ryzen node (AVX-512), up to 16 threads |
| Orca | OpenMP | AMD EPYC node (AVX2), up to 128 threads |
| Jutges | MPI | 16 nodes, 64 GB RAM each, up to 32 processes per node |

## Author

**Rafael Pino Rodríguez** · [LinkedIn](https://www.linkedin.com/in/rafael-pino-rodr%C3%ADguez-472918211)

Released under the [MIT License](LICENSE).
