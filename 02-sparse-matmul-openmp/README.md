# 02 · Sparse matrix multiplication with OpenMP

Product of two **8000 × 8000** matrices with 1 % non-zeros (640 000 each), stored as coordinate lists (`AD` sorted by rows, `BD` sorted by columns). The program computes the result in three ways and checks that they agree:

| Variant | Operation | Output |
|---------|-----------|--------|
| C1 | sparse × dense | dense |
| C2 | sparse × sparse | dense |
| CD | sparse × sparse | sparse (coordinate list) |

Expected output: `35248895` non-zeros in C, summing to `130625108751`. The benchmark script validates every run against these values.

## Parallelisation

The first attempt, a plain `parallel for` on the main loops, stalled at a speedup of about 2. Three bottlenecks explained it: sequential initialisation and sorting (Amdahl's law), cache thrashing from column-wise access to the dense matrix, and contention on the atomic counter used to build `CD`. The final version addresses each one:

- **Less sequential work.** Zero-initialisation of the dense matrices is done with `parallel for`. The two `qsort` calls run concurrently with `parallel sections`: one thread sorts `AD` while another generates and sorts `BD`.
- **Cache blocking for C1.** Columns are processed in blocks of 256 `int`s (a multiple of the 16 ints in a 64-byte cache line). Each thread runs through all the non-zeros of A against one block of B that stays in cache, with unit-stride access that the hardware prefetcher and the SIMD units can exploit.
- **Thread-local work vectors.** Each thread expands a column of B into its own dense vector `VBcol_local` and accumulates into `VCcol_local`. This avoids false sharing and write conflicts.
- **Dirty-index tracker.** Instead of resetting all 8000 positions of the result vector after every column, the code records which rows were touched (`idx_C`) and resets only those.
- **Bulk reservation for CD.** Each thread first counts the non-zeros of its column, then reserves that many consecutive slots in `CD` with **one** `atomic capture`, and fills them without further synchronisation. This replaces one atomic operation per element with one per column.
- Every multiplication of the original algorithm is still performed. The gains come from memory behaviour and synchronisation, not from skipping work.

## Results

Sequential baselines: 48.5 s on Orca, 16.2 s on Teen.

**Orca** (AMD EPYC)

| Threads | Time (s) | Speedup |
|--------:|---------:|--------:|
| 2 | 9.29 | 5.23 |
| 4 | 5.20 | 9.29 |
| 8 | 3.38 | 14.48 |
| 16 | 2.27 | 21.27 |
| 32 | 2.04 | **23.43** |
| 64 | 2.08 | 23.10 |
| 128 | 2.48 | 21.56 |

Harmonic mean **12.79** (target ≥ 5).

**Teen** (AMD Ryzen)

| Threads | Time (s) | Speedup |
|--------:|---------:|--------:|
| 2 | 6.29 | 2.57 |
| 4 | 3.75 | 4.32 |
| 8 | 2.35 | 6.90 |
| 16 | 1.63 | 9.94 |

Harmonic mean **4.62** (target ≥ 3.5).

Speedups above the thread count (for example 5.2× with 2 threads) come from the single-thread improvements: cache blocking, the dirty-index tracker and the cheaper synchronisation also speed up the code on one core. Beyond 32 threads the run is limited by memory bandwidth and by the sequential generation of `AD`, so adding threads no longer helps.

## Run

```bash
gcc -O3 -fopenmp spmm_omp.c -o spmm_omp      # static arrays: ~2.1 GB RAM
OMP_NUM_THREADS=8 ./spmm_omp

./run_benchmarks.sh      # on the URV cluster (SLURM partitions teen / orca)
```
