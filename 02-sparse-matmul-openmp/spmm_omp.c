/*
 * Sparse matrix multiplication (N = 8000, 1% density) parallelised with OpenMP.
 * Three variants: sparse x dense -> dense, sparse x sparse -> dense,
 * sparse x sparse -> sparse.
 * Computació Paral·lela i Massiva - Universitat Rovira i Virgili, 2025-26 (P1.2)
 * Author: Rafael Pino Rodríguez
 *
 * Build: gcc -O3 -fopenmp spmm_omp.c -o spmm_omp   (needs ~2.2 GB of RAM)
 * Run:   OMP_NUM_THREADS=8 ./spmm_omp
 */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <strings.h>
#include <assert.h>
#include <omp.h>

#define N 8000L
#define ND N*N/100
#define block_size 256

typedef struct {
    int i; //fila
    int j; //columna
    int v; //valor
} tmd;

int A[N][N],B[N][N],C[N][N],C1[N][N],C2[N][N];
int jBD[N+1],VCcol[N],VBcol[N];
tmd AD[ND],BD[ND],CD[N*N];

long long Suma;

int cmp_fil(const void *pa, const void *pb)
{
tmd * a = (tmd*)pa;
tmd * b = (tmd*)pb;

  if (a->i > b->i) return(1);
  else if (a->i < b->i) return (-1);
  else return (a->j - b->j);
}

int cmp_col(const void *pa, const void *pb)
{
tmd * a = (tmd*)pa;
tmd * b = (tmd*)pb;

  if (a->j > b->j) return(1);
  else if (a->j < b->j) return (-1);
  else return (a->i - b->i);
}

int main()
{
    int i,j,k,neleC;
// Parallel zero-initialisation of the dense matrices
    #pragma omp parallel for schedule(static)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = 0;
            C1[i][j] = 0;
            C2[i][j] = 0;
        }
    }

    // rand() is not thread-safe, so AD is generated sequentially
    for(k=0;k<ND;k++) {
        AD[k].i=rand()%(N-1);
        AD[k].j=rand()%(N-1);
        AD[k].v=rand()%100+1;
        while (A[AD[k].i][AD[k].j]) {
            if(AD[k].i < AD[k].j)
                AD[k].i = (AD[k].i + 1)%N;
            else 
                AD[k].j = (AD[k].j + 1)%N;
        }
        A[AD[k].i][AD[k].j] = AD[k].v;
    }

// Task parallelism: sort AD (by rows) while another thread generates and sorts BD (by columns)
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            qsort(AD,ND,sizeof(tmd),cmp_fil); // thread 1: sort AD by rows
        }
        #pragma omp section
        {
            for(int local_k=0; local_k<ND; local_k++) {
                BD[local_k].i=rand()%(N-1);
                BD[local_k].j=rand()%(N-1);
                BD[local_k].v=rand()%100+1;
                while (B[BD[local_k].i][BD[local_k].j]) {
                    if(BD[local_k].i < BD[local_k].j)
                        BD[local_k].i = (BD[local_k].i + 1)%N;
                    else 
                        BD[local_k].j = (BD[local_k].j + 1)%N;
                }
                B[BD[local_k].i][BD[local_k].j] = BD[local_k].v;
            }
            qsort(BD,ND,sizeof(tmd),cmp_col); // thread 2: generate BD and sort it by columns
        }
    }

    
  // Start index of every column of BD (CSC-like index)
    k=0;
    for (j=0; j<N+1; j++) {
      while (k < ND && j>BD[k].j) k++;
      jBD[j] = k;
    }

    // Original dense product, kept for reference:
    //for (i=0;i<N;i++)
    //    for (j=0;j<N;j++)
    //        for (k=0;k<N;k++)
    //            C[j][i] += A[j][k] * B[k][i];
    // Note: every 'omp for' below ends with an implicit barrier
    #pragma omp parallel default(none) shared(B, C1, C2, CD, AD, BD, jBD, neleC) private(i, j, k)
    {
        // Per-thread dense work vectors (declared inside the parallel region, hence private)
        int VBcol_local[N];
        int VCcol_local[N];
        for(j=0; j<N; j++) 
            VBcol_local[j] = VCcol_local[j] = 0;
    
        // ==========================================================
        // 1) Sparse x dense -> dense (C1), with cache blocking over columns
        // ==========================================================
        #pragma omp for schedule(static)
        for (int i0 = 0; i0 < N; i0 += block_size) {
            int i_end = (i0 + block_size > N) ? N : (i0 + block_size); // block of block_size columns
            // Every multiplication is still performed (no algorithmic shortcuts)
            for (k = 0; k < ND; k++) {
                int f = AD[k].i;
                int c = AD[k].j;
                int v = AD[k].v;
                
                // Unit-stride access over the block
                for (i = i0; i < i_end; i++) {
                    C1[f][i] += v * B[c][i];
                }
            }
        }
        //=========================================================
        // 2) Sparse x sparse -> dense (C2)
        //=========================================================

        #pragma omp for schedule(static) 
        for(i=0;i<N;i++) {
            // Expand column i of B into a dense vector
            for (k=jBD[i];k<jBD[i+1];k++)
                VBcol_local[BD[k].i] = BD[k].v;
            // All multiplications are performed, accumulating into thread-local vectors.
            // 'Dirty row' tracker: only touched entries are written back and reset.
            int idx_C[N];
            int n_idx = 0;

            for (k=0;k<ND;k++) {
                int f = AD[k].i;
                int mult = AD[k].v * VBcol_local[AD[k].j]; // always multiply
                if (mult > 0) {
                    if (VCcol_local[f] == 0) {
                        idx_C[n_idx++] = f;
                    }
                    VCcol_local[f] += mult;
                }
            }

            // Write back and reset only the dirty rows
            for (int x = 0; x < n_idx; x++) {
                int fila_sucia = idx_C[x];
                C2[fila_sucia][i] = VCcol_local[fila_sucia];
                VCcol_local[fila_sucia] = 0; 
            }
            
            for (k=jBD[i];k<jBD[i+1];k++)
                VBcol_local[BD[k].i] = 0;
        }

        #pragma omp single // one thread resets the shared counter
        {
            neleC=0;
        }
         
        //=========================================================
        // 3) Sparse x sparse -> sparse (CD)
        //=========================================================
        #pragma omp for schedule(static) 
        for(i=0;i<N;i++) {
            // Expand column i of B into a dense vector
            for (k=jBD[i];k<jBD[i+1];k++) 
                VBcol_local[BD[k].i] = BD[k].v;

            int idx_C[N];
            int n_idx = 0;

            for (k=0;k<ND;k++) {
                int fila = AD[k].i;
                int mult = AD[k].v * VBcol_local[AD[k].j];
                if (mult > 0) {
                    if (VCcol_local[fila] == 0) {
                        idx_C[n_idx++] = fila;
                    }
                    VCcol_local[fila] += mult;
                }
            }
            
            for (k=jBD[i];k<jBD[i+1];k++) 
                VBcol_local[BD[k].i] = 0;
            
           
            // Reserve n_idx consecutive slots in CD with a single atomic operation
            int local_neleC = 0;
            if (n_idx > 0) {
                #pragma omp atomic capture
                {
                    local_neleC = neleC;    
                    neleC += n_idx;        
                }
            }

            // Copy the non-zeros into the reserved slots, visiting only dirty rows
            for (int x = 0; x < n_idx; x++) {
                int fila_sucia = idx_C[x];
                CD[local_neleC].i = fila_sucia;
                CD[local_neleC].j = i;
                CD[local_neleC].v = VCcol_local[fila_sucia];
                VCcol_local[fila_sucia] = 0; 
                local_neleC++; 
            }
        }


    } // end of parallel region

    //=========================================================
    // Check: C1 (sparse x dense) == C2 (sparse x sparse)
    //=========================================================
    #pragma omp parallel for schedule(static)
    for (i=0;i<N;i++) {
        for(j=0;j<N;j++) {
            if (C2[i][j] != C1[i][j]) {
                printf("Diferencies C1 i C2 pos %d,%d: %d != %d\n",i,j,C1[i][j],C2[i][j]);
               }
           }
     }
    //=========================================================
    // Check: CD (sparse result) == C1, and sum of the values of CD
    //=========================================================
    Suma = 0;
    #pragma omp parallel for reduction(+:Suma) schedule(static)
    for(k=0;k<neleC;k++) {
         Suma += CD[k].v;
         if (CD[k].v != C1[CD[k].i][CD[k].j]) {
             printf("Diferencies C1 i CD a i:%d,j:%d,v%d, k:%d, vd:%d\n",CD[k].i,CD[k].j,C1[CD[k].i][CD[k].j],k,CD[k].v);
          }
     }
            
     printf ("\nNumero elements de la matriu dispersa C %d\n",neleC);
     printf("Suma dels elements de C %lld \n",Suma);
     exit(0);

}

