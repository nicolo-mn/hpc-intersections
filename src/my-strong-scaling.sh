#!/bin/sh

PROG=./omp-circles

if [ ! -f "$PROG" ]; then
    echo
    echo "Non trovo il programma $PROG."
    echo
    exit 1
fi

echo "p\tt1\tt2\tt3\tt4\tt5"

PROB_SIZE=10000 # default problem size
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores

for p in `seq $CORES`; do
    echo -n "$p\t"
    for rep in `seq 5`; do
        EXEC_TIME="$( OMP_NUM_THREADS=$p "$PROG" $PROB_SIZE | grep "Elapsed time:" | sed 's/Elapsed time: //' )"
        echo -n "${EXEC_TIME}\t"
    done
    echo ""
done
