#!/bin/bash
export LC_NUMERIC=C
cd "$(dirname "$0")"

# Benchmark on the URV SLURM partitions 'teen' and 'orca'.
gcc -O3 -fopenmp spmm_omp.c -o spmm_omp

# Sequential baselines given in the assignment (gcc -O3, N = 8000)

SEQ_TEEN=16.2
SEQ_ORCA=48.5

# Expected output of the sequential version (every run is validated against it)
EXPECTED_ELEMS="35248895"
EXPECTED_SUM="130625108751"

# Formato de las columnas para printf
ROW_FMT="%-13s | %-17s | %-22s | %-28s |\n"

echo "========================================================="
echo " PRUEBAS EN TEEN (Tiempo secuencial base: 16.2s)"
echo "========================================================="
sum_inv_S_teen=0
count_teen=0

printf "$ROW_FMT" "Threads" "Time (s)" "Speedup" "OK?"

for c in 2 4 8 16; do
    export OMP_NUM_THREADS=$c
    # Ejecutamos en silencio y lo guardamos en OUTPUT
    OUTPUT=$(srun -p teen -c $c time ./spmm_omp 2>&1)
    
    # Extraemos el valor pegado a "elapsed" ("0:14.90") y quitamos la palabra
    TIME_STR=$(echo "$OUTPUT" | grep -oE '[0-9]+:[0-9]+\.[0-9]+elapsed|[0-9]+\.[0-9]+elapsed' | sed 's/elapsed//')
    
    # Pasamos de minutos:segundos a segundos exactos (ej. 0:14.90 -> 14.90)
    TIME_PAR=$(echo "$TIME_STR" | awk -F: '{ if (NF==2) print ($1*60)+$2; else print $1 }')
    
    # Comprobamos si los resultados coinciden con los correctos
    if echo "$OUTPUT" | grep -q "$EXPECTED_ELEMS" && echo "$OUTPUT" | grep -q "$EXPECTED_SUM"; then
        RES="CORRECTO (Matriz validada)"
    else
        RES="ERROR (Resultados incorrectos)"
    fi
    
    # Calculamos el Speedup exacto
    SPEEDUP=$(awk -v seq=$SEQ_TEEN -v par=$TIME_PAR 'BEGIN { print seq/par }')
    
    # Imprimimos la fila entera perfectamente alineada
    printf "$ROW_FMT" "Threads: $c" "Time: $TIME_STR s" "Speedup: ${SPEEDUP}" "$RES"
    
    # Acumulamos para la media armónica: sum(1/Speedup) = sum(Par/Seq)
    sum_inv_S_teen=$(awk -v sum=$sum_inv_S_teen -v seq=$SEQ_TEEN -v par=$TIME_PAR 'BEGIN { print sum + (par/seq) }')
    ((count_teen++))
done

# Cálculo final Media Armónica Teen exacto
HMEAN_TEEN=$(awk -v n=$count_teen -v sum_inv=$sum_inv_S_teen 'BEGIN { print n/sum_inv }')
echo "---------------------------------------------------------"
echo "-> MEDIA ARMÓNICA OBTENIDA EN TEEN: $HMEAN_TEEN"
echo "========================================================="
echo ""


echo "========================================================="
echo " PRUEBAS EN ORCA (Tiempo secuencial base: 48.5s)"
echo "========================================================="
sum_inv_S_orca=0
count_orca=0

printf "$ROW_FMT" "Threads" "Time (s)" "Speedup" "OK?"

for c in 2 4 8 16 32 64 128; do
    export OMP_NUM_THREADS=$c
    OUTPUT=$(srun -p orca -c $c time ./spmm_omp 2>&1)
    
    TIME_STR=$(echo "$OUTPUT" | grep -oE '[0-9]+:[0-9]+\.[0-9]+elapsed|[0-9]+\.[0-9]+elapsed' | sed 's/elapsed//')
    TIME_PAR=$(echo "$TIME_STR" | awk -F: '{ if (NF==2) print ($1*60)+$2; else print $1 }')
    
    if echo "$OUTPUT" | grep -q "$EXPECTED_ELEMS" && echo "$OUTPUT" | grep -q "$EXPECTED_SUM"; then
        RES="CORRECTO (Matriz validada)"
    else
        RES="ERROR (Resultados incorrectos)"
    fi
    
    # Calculamos el Speedup exacto
    SPEEDUP=$(awk -v seq=$SEQ_ORCA -v par=$TIME_PAR 'BEGIN { print seq/par }')
    
    # Imprimimos la fila entera perfectamente alineada
    printf "$ROW_FMT" "Threads: $c" "Time: $TIME_STR s" "Speedup: ${SPEEDUP}" "$RES"
    
    sum_inv_S_orca=$(awk -v sum=$sum_inv_S_orca -v seq=$SEQ_ORCA -v par=$TIME_PAR 'BEGIN { print sum + (par/seq) }')
    ((count_orca++))
done

# Cálculo final Media Armónica Orca exacto
HMEAN_ORCA=$(awk -v n=$count_orca -v sum_inv=$sum_inv_S_orca 'BEGIN { print n/sum_inv }')
echo "---------------------------------------------------------"
echo "-> MEDIA ARMÓNICA OBTENIDA EN ORCA: $HMEAN_ORCA"
echo "========================================================="
rm -f spmm_omp
