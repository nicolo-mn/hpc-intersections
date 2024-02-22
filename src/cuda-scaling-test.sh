#!/bin/sh
export OMP_NUM_THREADS=12

echo "cuda\tomp"

for it in 5000 10000 15000 20000 30000 40000 50000 60000; do
    echo -n "$it\t"
    echo -n "$( ./cuda-circles $it 20 | grep "Elapsed time:" | sed 's/Elapsed time: //' )\t"
    echo "$( ./omp-circles $it 20 | grep "Elapsed time:" | sed 's/Elapsed time: //' )"
done