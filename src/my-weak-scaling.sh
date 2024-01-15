#!/bin/sh

# Questo script esegue il programma omp-matmul sfruttando OpenMP con
# un numero di core da 1 al numero di core disponibili sulla macchina
# (estremi inclusi). Il test con p processori viene effettuato su un
# input che ha dimensione N0 * (p^(1/3)), dove N0 e' la dimensione
# dell'input con p=1 thread OpenMP.
#
# Per come è stato implementato il programma parallelo, questo
# significa che all'aumentare del numero p di thread OpenMP, la
# dimensione del problema viene fatta crescere in modo che la quantità
# di lavoro per thread resti costante.
#
#-----------------------------------------------------------------------
# ATTENZIONE: il calcolo sopra vale solo per il programma omp-matmul;
# in generale non è detto che la dimensione dell'input debba crescere
# come la radice cubica del numero di core. Il calcolo andrà
# modificato in base all'effettivo costo asintotico dell'algoritmo da
# misurare, come spiegato a lezione.
#-----------------------------------------------------------------------

# Ultimo aggiornamento 2023-10-04
# Moreno Marzolla (moreno.marzolla@unibo.it)

PROG=./omp-matmul

if [ ! -f "$PROG" ]; then
    echo
    echo "Non trovo il programma $PROG."
    echo
    exit 1
fi

echo "p\tt1\tt2\tt3\tt4\tt5"

N0=1024 # base problem size
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores

for p in `seq $CORES`; do
    echo -n "$p\t"
    # Il comando bc non è in grado di valutare direttamente una radice
    # cubica, che dobbiamo quindi calcolare mediante logaritmo ed
    # esponenziale. L'espressione ($N0 * e(l($p)/3)) calcola
    # $N0*($p^(1/3))
    PROB_SIZE=`echo "$N0 * e(l($p)/3)" | bc -l -q`
    for rep in `seq 5`; do
        EXEC_TIME="$( OMP_NUM_THREADS=$p "$PROG" $PROB_SIZE | grep "Execution time" | sed 's/Execution time //' )"
        echo -n "${EXEC_TIME}\t"
    done
    echo ""
done
