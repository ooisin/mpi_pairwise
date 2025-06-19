# All Pairwise Computation problem using MPI 
Efficient distributed algorithm for all pairwise computations using MPI with a logical 1D ring topology and loop unrolling with sequential implementation for benchmark and correctness verification.

# Usage
## Single process (sequential only)
```bash
mpirun -np 1 ./pairwise <rows> <cols> <datatype>
```

## Multiple processes (parallel with sequential verification)
```bash
mpirun -np <num_processes> ./pairwise <rows> <cols> <datatype>
```

## Input args
rows: Positive integer ```N```
cols: Positive integer ```M```
datatype: ```float``` or ```double``` only - case sensitive
num_processes: Number of MPI processes - strictly positive

## Examples
```bash
mpirun -np 1 ./pairwise 512 256 double
mpirun -np 4 ./pairwise 1024 512 float
mpirun -np 8 ./pairwise 2048 1024 double
```
# Output
## Single Process Mode
```
Performing sequential computation on A: 1024 rows, 512 cols
Sequential computation time: 0.125000 seconds
Peak memory usage: 45.250000 MB
```

## Multi Process Mode
```
Performing sequential computation on A: 1024 rows, 512 cols
Sequential computation time: 0.125000 seconds
Performing parallel computation on A: 1024 rows, 512 cols
Parallel computation time: 0.045000 seconds
Peak memory usage: 52.125000 MB
Verification: PASSED
```

## Benchmark Results
The benchmark script generates a CSV file with metrics:

```csv
DataType,MatrixSize,Processes,SequentialTime,ParallelTime,Speedup,Efficiency,PeakMemoryMB,Verified
float,1024,4,0.125000,0.045000,2.7778,0.6944,52.125000,PASSED
double,2048,8,1.250000,0.180000,6.9444,0.8681,198.750000,PASSED
```

## CSV Columns
- **DataType**: `float` or `double`
- **MatrixSize**: Matrix dimension (M×M result from M×N input)
- **Processes**: Number of MPI processes
- **SequentialTime**: Sequential execution time in seconds
- **ParallelTime**: Parallel execution time in seconds  
- **Speedup**: SequentialTime / ParallelTime
- **Efficiency**: Speedup / Processes
- **PeakMemoryMB**: Peak RSS memory usage in megabytes
- **Verified**: `PASSED`/`FAILED` correctness check

