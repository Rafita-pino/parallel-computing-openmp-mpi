/*
 * K-means (1-D, N = 600000 points, G = 200 clusters) parallelised with OpenMP.
 * Computació Paral·lela i Massiva - Universitat Rovira i Virgili, 2025-26 (P1.1)
 * Author: Rafael Pino Rodríguez
 *
 * Build: gcc -O3 -fopenmp kmeans_omp.c -o kmeans_omp
 * Run:   OMP_NUM_THREADS=8 ./kmeans_omp
 */
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define N 600000
#define G 200

long V[N];
long R[G];
int A[G];

void kmean(int fN, int fK, long fV[], long fR[], int fA[]) {
    int iter = 0;
    long dif_global;
    long fS[G];
    int fD[N];

#pragma omp parallel default(none) shared(V, R, A, fD, fS, iter, dif_global)
    {
        int i, j;
        long t;

        do {
            #pragma omp single
            {
                for (i = 0; i < G; i++) {
                    fS[i] = 0;
                    A[i] = 0;
                }
            } 

            // =====================================================================
            // Assignment step: manual x8 unrolling (8 longs = one 64-byte cache
            // line) and labs() for native 64-bit absolute values.
            // nowait: with schedule(static) each thread reads back exactly the
            // fD entries it wrote, so no barrier is needed before the next loop.
            // =====================================================================
            #pragma omp for schedule(static) nowait
            for (i = 0; i < N; i += 8) {
                long v0 = V[i],   v1 = V[i+1], v2 = V[i+2], v3 = V[i+3];
                long v4 = V[i+4], v5 = V[i+5], v6 = V[i+6], v7 = V[i+7];
                
                long r0 = R[0];
                //dif = abs(fV[i] -fR[0]);
                long d0 = labs(v0 - r0), d1 = labs(v1 - r0), d2 = labs(v2 - r0), d3 = labs(v3 - r0);
                long d4 = labs(v4 - r0), d5 = labs(v5 - r0), d6 = labs(v6 - r0), d7 = labs(v7 - r0);
                
                //min = 0;
                int m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0, m5 = 0, m6 = 0, m7 = 0;
                
                for (j = 1; j < G; j++) {
                    long r = R[j]; 
                    //dif = abs(fV[i] -fR[0]);
                    long c0 = labs(v0 - r), c1 = labs(v1 - r), c2 = labs(v2 - r), c3 = labs(v3 - r);
                    long c4 = labs(v4 - r), c5 = labs(v5 - r), c6 = labs(v6 - r), c7 = labs(v7 - r);
                    
                    //if (abs(fV[i] -fR[j]) < dif)
                    if (c0 < d0) { d0 = c0; m0 = j; }
                    if (c1 < d1) { d1 = c1; m1 = j; }
                    if (c2 < d2) { d2 = c2; m2 = j; }
                    if (c3 < d3) { d3 = c3; m3 = j; }
                    if (c4 < d4) { d4 = c4; m4 = j; }
                    if (c5 < d5) { d5 = c5; m5 = j; }
                    if (c6 < d6) { d6 = c6; m6 = j; }
                    if (c7 < d7) { d7 = c7; m7 = j; }
                }
                
                //fD[i] = min;
                fD[i]   = m0; fD[i+1] = m1; fD[i+2] = m2; fD[i+3] = m3;
                fD[i+4] = m4; fD[i+5] = m5; fD[i+6] = m6; fD[i+7] = m7;
            }

            // Update step: array-section reduction, private copies of fS/A (~2.4 KB) stay in L1
            #pragma omp for reduction(+:fS[:G], A[:G]) schedule(static)
            for (i = 0; i < N; i += 8) {
                int idx0 = fD[i],   idx1 = fD[i+1], idx2 = fD[i+2], idx3 = fD[i+3];
                int idx4 = fD[i+4], idx5 = fD[i+5], idx6 = fD[i+6], idx7 = fD[i+7];
                
                //fS[fD[i]] += fV[i];
                fS[idx0] += V[i];   fS[idx1] += V[i+1]; fS[idx2] += V[i+2]; fS[idx3] += V[i+3];
                fS[idx4] += V[i+4]; fS[idx5] += V[i+5]; fS[idx6] += V[i+6]; fS[idx7] += V[i+7];
                
                //fA[fD[i]]++;
                A[idx0]++; A[idx1]++; A[idx2]++; A[idx3]++;
                A[idx4]++; A[idx5]++; A[idx6]++; A[idx7]++;
            }

            #pragma omp single
            {
                dif_global = 0;
                for (i = 0; i < G; i++) {
                    t = R[i];
                    if (A[i]) {
                        R[i] = fS[i] / A[i];
                    }
                    dif_global += labs(t - R[i]);
                }
                iter++;
            }
        } while (dif_global);
    }
    
    printf("iter %d\n", iter);
}

void qs(int ii, int fi, long fV[], int fA[]){
    int i, f;
    long pi, pa, vtmp, vta, vfi, vfa;

    pi = fV[ii];
    pa = fA[ii];
    i = ii + 1;
    f = fi;
    vtmp = fV[i];
    vta = fA[i];

    while (i <= f) {
        if (vtmp < pi) {
            fV[i - 1] = vtmp;
            fA[i - 1] = vta;
            i++;
            vtmp = fV[i];
            vta = fA[i];
        } else {
            vfi = fV[f];
            vfa = fA[f];
            fV[f] = vtmp;
            fA[f] = vta;
            f--;
            vtmp = vfi;
            vta = vfa;
        }
    }
    fV[i - 1] = pi;
    fA[i - 1] = pa;

    if (ii < f) qs(ii, f, fV, fA);
    if (i < fi) qs(i, fi, fV, fA);
}

int main(){
    int i;

    // Sequential initialisation: rand() is not thread-safe
    for (i = 0; i < N; i++) {
        V[i] = (rand() % rand()) / N;
    }

    for (i = 0; i < G; i++) {
        R[i] = V[i];
    }

    kmean(N, G, V, R, A);

    qs(0, G - 1, R, A);

    for (i = 0; i < G; i++) {
        printf("R[%d] : %ld te %d agrupats\n", i, R[i], A[i]);
    }

    return 0;
}
