#!/bin/bash

set -e

# test mode parameter
TEST_MODE=false
if [[ "$1" == "test" ]]; then
    TEST_MODE=true
    echo "Running in TEST MODE: A=1024x1024 float with p=2,4,8,16"
fi

# CSV file in current directory
CSV_FILE="benchmark_results.csv"

# Init the result CSV headers
echo "DataType,MatrixSize,Processes,SequentialTime,ParallelTime,Speedup,Efficiency,PeakMemoryMB,Verified" > "$CSV_FILE"

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
    if ! mpirun -np "$num_procs" --oversubscribe ./pairwise "$matrix_size" "$matrix_size" "$datatype" > "$temp_output" 2>&1; then
        echo "ERROR: MPI run failed for $datatype ${matrix_size}x${matrix_size} with $num_procs processes"
        rm -f "$temp_output" "$temp_seq_output"
        return 1
    fi
    
    # Extract timing information
    local seq_time=$(grep "Sequential computation time:" "$temp_output" | awk '{print $4}')
    local par_time=$(grep "Parallel computation time:" "$temp_output" | awk '{print $4}')
    
    # For single process, use sequential time as parallel time
    if [[ -z "$par_time" ]]; then
        par_time="$seq_time"
    fi
    
    # Extract memory usage
    local memory_usage=$(grep "Peak memory usage:" "$temp_output" | awk '{print $4}')
    if [[ -z "$memory_usage" ]]; then
        memory_usage="N/A"
    fi
    
    # Extract verification status
    local verified=$(grep "Verification:" "$temp_output" | awk '{print $2}')
    if [[ -z "$verified" ]]; then
        verified="N/A" # Doesnt verify when single proc
    fi
    
    # speedup and efficiency
    local speedup
    local efficiency
    if (( $(echo "$par_time > 0" | bc -l) )); then
        speedup=$(echo "scale=6; $seq_time / $par_time" | bc -l)
        efficiency=$(echo "scale=6; $speedup / $num_procs" | bc -l)
    else
        speedup="N/A"
        efficiency="N/A"
    fi

    # Log results to CSV
    echo "$datatype,$matrix_size,$num_procs,$seq_time,$par_time,$speedup,$efficiency,$memory_usage,$verified" >> "$CSV_FILE"
    
    # Display results
    printf "  Sequential Time: %8.6f seconds\n" "$seq_time"
    printf "  Parallel Time:   %8.6f seconds\n" "$par_time"
    printf "  Speedup:         %8.4f\n" "$speedup"
    printf "  Efficiency:      %8.4f\n" "$efficiency"
    printf "  Peak Memory:     %8.2f MB\n" "$memory_usage"
    printf "  Verification:    %s\n" "$verified"
    echo ""
    
    rm -f "$temp_output" "$temp_seq_output"
}

# params
if [[ "$TEST_MODE" == "true" ]]; then
    DATATYPES=("float")
    MATRIX_SIZES=(1024)
    PROCESS_COUNTS=(2 4)
else
    DATATYPES=("float" "double")
    MATRIX_SIZES=(256 512 1024 2048)
    PROCESS_COUNTS=(1 2 4 8 16)
fi

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

echo "Benchmark Complete"
echo "Total tests: $total_tests"
echo "Completed: $completed_tests"
echo "Failed: $failed_tests"
echo ""
echo "Benchmark results written to: $CSV_FILE"
