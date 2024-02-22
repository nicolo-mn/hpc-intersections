#!/bin/sh

PROG=./omp-circles

if [ ! -f "$PROG" ]; then
    echo
    echo "Non trovo il programma $PROG."
    echo
    exit 1
fi

echo "size\tt1\tt2\tt3\tt4\tt5"

PROB_SIZE=10000 # default problem size
DIM=256

for d in `seq $DIM`; do
    echo -n "$d\t"
    for rep in `seq 5`; do
        EXEC_TIME="$( OMP_SCHEDULE="dynamic,$d" "$PROG" $PROB_SIZE | grep "Elapsed time" | sed 's/Elapsed time: //' )"
        echo -n "${EXEC_TIME}\t"
    done
    echo ""
done
