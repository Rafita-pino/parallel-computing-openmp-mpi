# 03 · K-means with MPI

1-D K-means clustering of **2 000 000** values into **400** clusters, distributed across up to 16 nodes. Requirements of the assignment: same output as the sequential version, the final result stored in rank 0, harmonic-mean speedup over the 9 configurations > 4, best speedup > 15 and worst speedup > 1.75. **All of them were met.**

## Parallelisation

**Domain decomposition.** The points are split into contiguous blocks. The remainder `N % p` is spread across the first ranks, so block sizes differ by at most one.

**Shadow generation instead of `Scatter`.** In the first version, rank 0 generated all 2 M points and sent them with `MPI_Scatterv`. Every other rank sat idle meanwhile, and rank 0's network link became a bottleneck. In the final version, every rank seeds `rand()` with the same value, runs through the same sequence and stores only its own slice. The input is distributed without sending a single byte.

**Decentralised reduction.** Each iteration:

1. Every rank assigns its local points to the nearest centroid.
2. It accumulates the sum and the count per cluster into a packed buffer, with the sum in even positions and the count in odd ones. Sum and count of a cluster share a cache line, and one buffer means one message.
3. A single `MPI_Allreduce` combines the buffers. Every rank receives the global totals and recomputes the centroids itself. This removes both the master-side division step and the `MPI_Bcast` of the earlier `Reduce` + `Bcast` design.

The local buffers are aligned to 64 bytes (`__attribute__((aligned(64)))`), the L1 cache line size reported by the nodes, to avoid cache-line splits. Rank 0 prints the result through a single buffered `printf`.

## Results

Sequential baseline: 26.2 s. "Time" is the wall-clock time of `srun`. *MPI_Wtime* measures only the algorithm, excluding process start-up.

| Nodes × processes/node | Processes | Time (s) | Speedup | Speedup (MPI_Wtime) |
|------------------------|----------:|---------:|--------:|--------------------:|
| 2 × 1 | 2 | 12.084 | 2.17 | 2.27 |
| 4 × 1 | 4 | 6.419 | 4.08 | 4.46 |
| 8 × 1 | 8 | 3.763 | 6.96 | 8.03 |
| 16 × 1 | 16 | 2.259 | 11.60 | 15.17 |
| 16 × 2 | 32 | 1.809 | 14.48 | 23.48 |
| 16 × 4 | 64 | 1.358 | 19.29 | 35.04 |
| 16 × 8 | 128 | 1.348 | **19.44** | **39.70** |
| 16 × 16 | 256 | 1.485 | 17.64 | 36.53 |
| 16 × 32 | 512 | 7.097 | 3.69 | 4.21 |

Harmonic mean: **6.27** (wall clock) and **7.39** (MPI_Wtime).

At 16 × 32 the nodes are oversubscribed (`--overcommit`, more processes than cores), so processes compete for CPU time and every `Allreduce` waits for the slowest one. The difference between the two speedup columns shows how much of the wall-clock time at high process counts is process start-up rather than computation.

## Run

```bash
mpicc -O3 kmeans_mpi.c -o kmeans_mpi
mpirun -np 4 ./kmeans_mpi

./run_benchmarks.sh      # on the URV cluster (SLURM partition jutjat, 16 nodes)
```

The benchmark script first runs one process to build a reference output, then compares every configuration against it line by line.
