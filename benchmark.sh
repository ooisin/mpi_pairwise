#!/bin/bash

set -e

# results dir
RESULTS_DIR="benchmark_results"
CSV_FILE="$RESULTS_DIR/benchmark_results.csv"

mkdir -p "$RESULTS_DIR"

# Init the result CSV headers
echo "DataType,MatrixSize,Processes,SequentialTime,ParallelTime,Speedup,Verified" > "$CSV_FILE"

make clean
make

run_benchmark() {
    local datatype=$1
    local matrix_size=$2
    local num_procs=$3
    
    echo "Testing: DataType=$datatype, Size=${matrix_size}x${matrix_size}, Processes=$num_procs"
    
    # Create temp files for output parsing
    # Note: this is specific/fragile parser -> needs to be exact!
    local temp_output=$(mktemp)
    local temp_seq_output=$(mktemp)
    
    # Run MPI
    if ! mpirun -np "$num_procs" ./pairwise "$matrix_size" "$matrix_size" "$datatype" > "$temp_output" 2>&1; then
        echo "ERROR: MPI run failed for $datatype ${matrix_size}x${matrix_size} with $num_procs processes"
        rm -f "$temp_output" "$temp_seq_output"
        return 1
    fi
    
    # Extract timing information
    local seq_time=$(grep "Sequential computation time:" "$temp_output" | awk '{print $4}')
    local par_time=$(grep "Computation time:" "$temp_output" | tail -1 | awk '{print $3}')
    
    # speedup
    local speedup
    if (( $(echo "$par_time > 0" | bc -l) )); then
        speedup=$(echo "scale=6; $seq_time / $par_time" | bc -l)
    else
        speedup="N/A"
    fi
    
    # TODO: Print a pass status from program on sequential verification

    # Log results to CSV
    echo "$datatype,$matrix_size,$num_procs,$seq_time,$par_time,$speedup,$verified" >> "$CSV_FILE"
    
    # Display results
    printf "  Sequential Time: %8.6f seconds\n" "$seq_time"
    printf "  Parallel Time:   %8.6f seconds\n" "$par_time"
    printf "  Speedup:         %8.4f\n" "$speedup"
    printf "  Verification:    %s\n" "$verified"
    echo ""
    
    rm -f "$temp_output" "$temp_seq_output"
}

# params
DATATYPES=("float" "double")
MATRIX_SIZES=(256 512 1024 2048)
PROCESS_COUNTS=(1 2 4 8 16)

total_tests=0
completed_tests=0
failed_tests=0

# Run all configs of benchmarks
for datatype in "${DATATYPES[@]}"; do
    for matrix_size in "${MATRIX_SIZES[@]}"; do
        for num_procs in "${PROCESS_COUNTS[@]}"; do
            total_tests=$((total_tests + 1))
            
            echo "[$completed_tests/$total_tests] Running benchmark: "
            
            if run_benchmark "$datatype" "$matrix_size" "$num_procs"; then
                completed_tests=$((completed_tests + 1))
            else
                failed_tests=$((failed_tests + 1))
                echo "FAILED: $datatype ${matrix_size}x${matrix_size} with $num_procs processes"
                echo ""
            fi
        done
    done
done

echo "__|^|  Benchmark Complete |^|___"
echo "Total tests: $total_tests"
echo "Completed: $completed_tests"
echo "Failed: $failed_tests"
echo ""
echo "Benchmark results written to: $CSV_FILE"
