# 04 · Sparse matrix multiplication with MPI

The same three products as in [project 02](../02-sparse-matmul-openmp/) (sparse × dense → dense, sparse × sparse → dense, sparse × sparse → sparse), now distributed across up to 16 nodes. The matrix size `N` is a command-line argument and the matrices live in dynamic memory. With `N = 8000` the result must match the sequential one: `35248895` non-zeros summing to `130625108751`.

One process with `N = 8000` uses about 2 GB, and a Jutges node has 64 GB. Running 32 processes per node therefore needs careful memory use.

## Parallelisation

**Column-block decomposition.** Each rank owns a contiguous block of columns of the result and allocates only its share of `C`, `C1` and `C2` (`N × N_local`). Columns were chosen because the final compression into `CD` is naturally done column by column.

**Shadow generation.** As in project 03, every rank rebuilds the sparse inputs `AD` and `BD` from the same seed, so nothing needs to be sent at start-up.

**Cache-friendly kernels.**
- *Loop interchange* in the sparse × dense product: the loop over non-zeros is the outer one and the loop over the local columns the inner one, so each rank writes its slice of `C1` with unit stride.
- *Sparse reset*: after each column, only the entries of the work vector that were set from `BD` are cleared, using the `jBD` column index, instead of all `N` positions.
- *Alignment*: all buffers are allocated with `posix_memalign(…, 64, …)`. The 12-byte `tmd` struct is padded to 16 bytes with `__attribute__((aligned(16)))`, so exactly four fit in a 64-byte cache line and none straddles two lines.

**One collective.** Each rank checks its own columns (`C1 == C2 == CD`). Then a single `MPI_Reduce` of a packed array `{sum, nnz, multiplications}` sends the global totals to rank 0.

**Benchmark script.** It calls `salloc` once for 16 exclusive nodes and runs all nine configurations inside that allocation, with timing measured inside `srun`, so queueing and scheduler overhead don't distort the results.

## Results (N = 8000)

Sequential baseline: 19.5 s.

| Nodes × processes/node | Processes | Time (s) | Speedup |
|------------------------|----------:|---------:|--------:|
| 2 × 1 | 2 | 3.670 | 5.31 |
| 4 × 1 | 4 | 2.130 | 9.15 |
| 8 × 1 | 8 | 1.340 | 14.55 |
| 16 × 1 | 16 | 0.950 | 20.53 |
| 16 × 2 | 32 | 0.790 | **24.68** |
| 16 × 4 | 64 | 0.830 | 23.49 |
| 16 × 8 | 128 | 1.020 | 19.12 |
| 16 × 16 | 256 | 1.570 | 12.42 |
| 16 × 32 | 512 | 3.760 | 5.19 |

Harmonic mean: **10.93**. No minimum was required for this assignment; the requirement was a correct result in all nine configurations, which every run achieved.

The speedup above the process count at small scale (5.3× with 2 processes) comes from the memory-access improvements, which also help a single process. Beyond 32 processes every rank still regenerates and sorts the full sparse inputs, a fixed cost that doesn't shrink with more processes. On top of that, ranks on the same node compete for memory bandwidth, so performance falls again.

## Run

```bash
mpicc -O3 spmm_mpi.c -o spmm_mpi
mpirun -np 4 ./spmm_mpi 8000

./run_benchmarks.sh      # on the URV cluster (SLURM partition jutjat, 16 nodes)
```
