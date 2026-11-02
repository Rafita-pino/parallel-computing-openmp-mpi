# 01 · K-means with OpenMP

1-D K-means clustering of **600 000** values into **200** clusters, parallelised for shared memory. The parallel version must produce exactly the same clusters as the sequential one (`iter 147`, last centroid `3390` with `1878` points). The benchmark script checks this on every run.

## Dependence analysis

The algorithm repeats four steps until the centroids stop moving:

1. **Assignment**: find the nearest centroid for every point and store it in `fD`.
2. **Reset** the per-cluster accumulators `fS` (sum) and `A` (count).
3. **Accumulation**: add every point to its cluster's sum and count.
4. **Update**: recompute the centroids `R` and the total displacement.

The only loop-carried dependence is **4 → 1**: the next iteration needs the new centroids. Inside an iteration, step 1 is fully parallel over the points, and step 3 is a reduction. The input generation uses `rand()`, which isn't thread-safe, so it stays sequential.

## Parallelisation

A single `parallel` region wraps the whole `do … while` loop, so threads are created only once.

- **Assignment loop**: `omp for schedule(static) nowait`, manually **unrolled ×8**. Eight `long` values are exactly one 64-byte cache line, and keeping eight independent `min` searches in flight gives the CPU instruction-level parallelism that the `if` inside the loop would otherwise block.
- **`labs()` instead of `abs()`**: the original code called the 32-bit `abs()` on 64-bit values. Using `labs()` avoids the conversions.
- **`nowait`**: with a static schedule, each thread reads back in step 3 exactly the `fD` entries it wrote in step 1, so the barrier between the two loops is unnecessary.
- **Array-section reduction** `reduction(+:fS[:G], A[:G])`: each thread accumulates into private copies of about 2.4 KB that stay in L1 cache, instead of contending for shared counters.
- **Reset and update** run inside `omp single`, since they only touch 200 elements.

## Results

Sequential baselines: 17.3 s on Orca, 8.8 s on Teen.

**Orca** (AMD EPYC)

| Threads | Time (s) | Speedup |
|--------:|---------:|--------:|
| 2 | 9.44 | 1.83 |
| 4 | 4.74 | 3.64 |
| 8 | 2.40 | 7.21 |
| 16 | 1.31 | 13.20 |
| 32 | 0.65 | 26.61 |
| 64 | 0.36 | 48.05 |
| 128 | 0.24 | **72.08** |

Harmonic mean **6.32** (target ≥ 6).

**Teen** (AMD Ryzen)

| Threads | Time (s) | Speedup |
|--------:|---------:|--------:|
| 2 | 5.63 | 1.56 |
| 4 | 2.83 | 3.11 |
| 8 | 1.48 | 5.94 |
| 16 | 0.77 | 11.42 |

Harmonic mean **3.28** (target ≥ 4). Scaling is close to linear up to 16 threads. The harmonic mean is dominated by the 2-thread run, where the sequential parts and the fixed per-iteration overhead weigh the most.

## Run

```bash
gcc -O3 -fopenmp kmeans_omp.c -o kmeans_omp
OMP_NUM_THREADS=8 ./kmeans_omp

./run_benchmarks.sh      # on the URV cluster (SLURM partitions teen / orca)
```
