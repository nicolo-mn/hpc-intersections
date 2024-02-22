#!/bin/sh

PROG=./omp-circles

if [ ! -f "$PROG" ]; then
    echo
    echo "Non trovo il programma $PROG."
    echo
    exit 1
fi

echo "p\tt1\tt2\tt3\tt4\tt5"

N0=10000 # base problem size
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores

for p in `seq $CORES`; do
    echo -n "$p\t"
    # Il comando bc non è in grado di valutare direttamente una radice
    # cubica, che dobbiamo quindi calcolare mediante logaritmo ed
    # esponenziale. L'espressione ($N0 * e(l($p)/3)) calcola
    # $N0*($p^(1/3))
    PROB_SIZE=`echo "$N0 * e(l($p)/2)" | bc -l -q`
    for rep in `seq 5`; do
        EXEC_TIME="$( OMP_NUM_THREADS=$p "$PROG" $PROB_SIZE | grep "Elapsed time:" | sed 's/Elapsed time: //' )"
        echo -n "${EXEC_TIME}\t"
    done
    echo ""
done
