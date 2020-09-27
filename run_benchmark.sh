#!/bin/bash

set -e

SCRIPT_PATH=$(pwd)/$(dirname $0)
BENCHMARK_BIN=${SCRIPT_PATH}/bin/benchmark.o

TYPES=(
    "matrix_matrix"
    "matrix_matrix_cblas"
    "matrix_vector"
    "matrix_vector_cblas"
)

N_RANGE=(
    256 512 1024 2048 4096
)

for TYPE in ${TYPES[@]}; do

    for N in ${N_RANGE[@]}; do
        T1=$(date +%s.%N)

        ${BENCHMARK_BIN} ${TYPE} ${N}

        T2=$(date +%s.%N)

        SCORE=$(echo "${T2} - ${T1}" | bc)

        echo "${TYPE} N=${N} TIME: ${SCORE}"
    done

done
