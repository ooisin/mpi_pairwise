#include "matrix.h"
#include "mpi.h"
#include "validate.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

// Gets current peak RSS memory in MB for process.
double get_peak_memory()
{  
        // Convert KB to MB -> Linux assumption
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
                return usage.ru_maxrss / 1024.0;
        }
        return -1.0;
}

/**
 * Determines block size/width (number of columns) and handles the imperfect
 * distribution case M % p != 0 by giving some processes an extra column.
 * The first M % size procs get M/size + 1 cols, the remaining get M/size cols
 * to minimise the load imbalance across all processes fairly. Maintaining the
 * offset ensures no columns overlap aacross blocks.
 */
void calculate_block_size(int M, int size, int rank, int* block_size, int* offset)
{
        int base = M / size;
        int remainder = M % size;

        if (rank < remainder) {
                *block_size = base + 1;
                *offset = rank * (base + 1);
        } else {
                *block_size = base;
                *offset = remainder * (base + 1) + (rank - remainder) * base;
        }
}

/**
 * Computes dot products between local matrix rows and received matrix rows in the distributed pairwise algorithm.
 * Handles the core computation step of the ring algorithm processing one portion of dot products at each ring step -
 * only computing the upper triangle of the result matrix since dot products are symmetric. Loop unrolling with factor 4
 * improves performance by reducing loop overhead. IJK loop order is optimal in this scenario as the matrix is stored
 * row-major -> we dont access columns as we are working with an implicit tranpose of the input dimensions. Therefore,
 * pairwise row dot prodcut of A^T = pairwise col dot product of A. Both give M x M result matrix.
 */
void compute_dot_product(void** A_local, void** A_received, void** C_local, int N, int M, int local_block_rows,
                         int recv_block_rows, int rank, int step, int size, MPI_Datatype dtype)
{
        // Handle the type casts based on input
        if (dtype == MPI_DOUBLE) {
                double** A_loc = (double**)A_local;
                double** A_rec = (double**)A_received;
                double** C_loc = (double**)C_local;

                // Calculate the gloabl matrix pos for local and rec blocks
                // Calculate local offset for this rank - offset is the slice of the final result matrix the process
                // owns
                int local_offset;
                int temp_local_rows;
                calculate_block_size(M, size, rank, &temp_local_rows, &local_offset);

                // Calculate remote offset for sending rank
                int remote_rank = (rank - step + size) % size;
                int remote_offset;
                int temp_remote_rows;
                calculate_block_size(M, size, remote_rank, &temp_remote_rows, &remote_offset);

                // Process only upper triangle elements - avoid redundant symmetrical comps
                for (int i = 0; i < local_block_rows; i++) {
                        int global_i = local_offset + i;

                        for (int j = 0; j < recv_block_rows; j++) {
                                int global_j = remote_offset + j;

                                // Skip lower triangle computations
                                if (global_i > global_j) {
                                        continue;
                                }

                                // loop unrolling for dot product computation with factor 4
                                double A_k = 0.0;
                                double A_k1 = 0.0;
                                double A_k2 = 0.0;
                                double A_k3 = 0.0;

                                int k;

                                // Row major/based access: A_loc[i][k] gives element k of vector i
                                // This computes dot product between row i,j
                                for (k = 0; k < N - 3; k += 4) {
                                        A_k += A_loc[i][k] * A_rec[j][k];
                                        A_k1 += A_loc[i][k + 1] * A_rec[j][k + 1];
                                        A_k2 += A_loc[i][k + 2] * A_rec[j][k + 2];
                                        A_k3 += A_loc[i][k + 3] * A_rec[j][k + 3];
                                }

                                // Handle remaining elements
                                for (; k < N; k++) {
                                        A_k += A_loc[i][k] * A_rec[j][k];
                                }

                                // Store result in appropriate position of global matrix
                                double sum = A_k + A_k1 + A_k2 + A_k3;
                                C_loc[i][global_j] = sum;
                        }
                }
        } else if (dtype == MPI_FLOAT) {
                float** A_loc = (float**)A_local;
                float** A_rec = (float**)A_received;
                float** C_loc = (float**)C_local;

                int local_offset;
                int temp_local_rows;
                calculate_block_size(M, size, rank, &temp_local_rows, &local_offset);

                int remote_rank = (rank - step + size) % size;
                int remote_offset;
                int temp_remote_rows;
                calculate_block_size(M, size, remote_rank, &temp_remote_rows, &remote_offset);

                for (int i = 0; i < local_block_rows; i++) {
                        int global_i = local_offset + i;

                        for (int j = 0; j < recv_block_rows; j++) {
                                int global_j = remote_offset + j;

                                if (global_i > global_j) continue;

                                float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
                                int k;

                                // Row-based dot product computation
                                for (k = 0; k < N - 3; k += 4) {
                                        sum1 += A_loc[i][k] * A_rec[j][k];
                                        sum2 += A_loc[i][k + 1] * A_rec[j][k + 1];
                                        sum3 += A_loc[i][k + 2] * A_rec[j][k + 2];
                                        sum4 += A_loc[i][k + 3] * A_rec[j][k + 3];
                                }

                                for (; k < N; k++) {
                                        sum1 += A_loc[i][k] * A_rec[j][k];
                                }

                                float sum = sum1 + sum2 + sum3 + sum4;
                                C_loc[i][global_j] = sum;
                        }
                }
        }
}

/**
 * Implements the ring-based distributed pairwise dot product algorithm.
 * This function orchestrates the parallel computation using a 1D ring topology where each process exchanges matrix row
 * blocks with its neighbors. Every process gets to compute dot products with every other process data through a series
 * of cyclic exchanges which avoids expensive communication costs/patterns.
 */
void ring_pairwise(void** A_loc, void** C_loc, int N, int M, int rank, int size, MPI_Datatype type, size_t elem_size,
                   int local_block_rows)
{
        // Establishes the ring topology so each proc has left/prev and right/next neighbour establishing a circular
        // communication pattern
        int next_rank = (rank + 1) % size;
        int prev_rank = (rank + size - 1) % size;
        int max_block_rows = (M + size - 1) / size; // Max rows any process might have

        // Communication buffers for ring shifts
        void** send_buf = alloc_matrix(max_block_rows, N, elem_size, elem_size);
        void** recv_buf = alloc_matrix(max_block_rows, N, elem_size, elem_size);

        // Initialise send buffer with local block whcih is essentially the starting pos before any excnange - Each proc
        // starts with its own block first.
        for (int i = 0; i < local_block_rows; i++) {
                memcpy(send_buf[i], A_loc[i], N * elem_size);
        }
        int current_block_rows = local_block_rows;

        // After size number of steps (num procs) every proc will access the block/data of anohter proc once by the
        // ring/cyclic topology -> at step k, each proc will have data/rows that are from the proc that is k positions
        // to its left. This allows O(p) comm cost
        for (int step = 0; step < size; step++) {
                // Calculate block size for current step
                int sending_rank = (rank - step + size) % size;
                int step_block_rows, step_offset;
                calculate_block_size(M, size, sending_rank, &step_block_rows, &step_offset);

                compute_dot_product(A_loc, send_buf, C_loc, N, M, local_block_rows, step_block_rows, rank, step, size,
                                    type);

                // Ring shift with MPI_Sendrecv to avoid deadlocks
                if (step < size - 1) {
                        int next_step_rank = (rank - step - 1 + size) % size;
                        int next_block_rows, next_offset;
                        calculate_block_size(M, size, next_step_rank, &next_block_rows, &next_offset);

                        MPI_Sendrecv(send_buf[0], current_block_rows * N, type, next_rank, 0, recv_buf[0],
                                     next_block_rows * N, type, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        // Swap buffers for next iteration to reduce amount of data being copied and track the current
                        // new block size so we can handle cases when M % p != 0
                        void** temp = send_buf;
                        send_buf = recv_buf;
                        recv_buf = temp;
                        current_block_rows = next_block_rows;
                }
        }

        free(send_buf);
        free(recv_buf);
}

/**
 * Computes the sequential pairwise dot product matrix for all row vectors in A.
 * This serves as the reference implementation for correctness verification and the baseline for performance comparison
 * against the parallel implementation. The function computes the result C where each element C[i][j] represents the dot
 * product between row i and row j of A which is symmetric matrix of all pairwise computations.
 */
void sequential_dotprod(void** A, void** C, int rows, int cols, DataType type)
{
        // Handle type casting depdening on input
        if (type == DOUBLE_TYPE) {
                double** A_mat = (double**)A;
                double** C_mat = (double**)C;

                // Compute result mat C
                // Out loop iterates the rows of A
                for (int i = 0; i < rows; i++) {
                        // Only compute upper triangle
                        for (int j = 0; j <= i; j++) {
                                double A_k = 0.0;
                                double A_k1 = 0.0;
                                double A_k2 = 0.0;
                                double A_k3 = 0.0;

                                int k;

                                // Core dot product computation of row i,j with loop unrolling where each iter computes
                                // the 4 sums = reduced loop overhead
                                for (k = 0; k < cols - 3; k += 4) {
                                        A_k += A_mat[i][k] * A_mat[j][k];
                                        A_k1 += A_mat[i][k + 1] * A_mat[j][k + 1];
                                        A_k2 += A_mat[i][k + 2] * A_mat[j][k + 2];
                                        A_k3 += A_mat[i][k + 3] * A_mat[j][k + 3];
                                }

                                // Handle remaining elements i.e any remainder rows not divisible by 4
                                for (; k < cols; k++) {
                                        A_k += A_mat[i][k] * A_mat[j][k];
                                }

                                double sum = A_k + A_k1 + A_k2 + A_k3;
                                C_mat[j][i] = sum;
                        }
                }
        } else if (type == FLOAT_TYPE) {
                float** A_mat = (float**)A;
                float** C_mat = (float**)C;

                // Same implementation as double type
                for (int i = 0; i < rows; i++) {
                        // Only compute upper triangle
                        for (int j = 0; j <= i; j++) {
                                float A_k = 0.0;
                                float A_k1 = 0.0;
                                float A_k2 = 0.0;
                                float A_k3 = 0.0;

                                int k;

                                // Dot product of row i,j
                                for (k = 0; k < cols - 3; k += 4) {
                                        A_k += A_mat[i][k] * A_mat[j][k];
                                        A_k1 += A_mat[i][k + 1] * A_mat[j][k + 1];
                                        A_k2 += A_mat[i][k + 2] * A_mat[j][k + 2];
                                        A_k3 += A_mat[i][k + 3] * A_mat[j][k + 3];
                                }

                                // Handle remaining elements
                                for (; k < cols; k++) {
                                        A_k += A_mat[i][k] * A_mat[j][k];
                                }

                                // Store in both positions by symmetry
                                float sum = A_k + A_k1 + A_k2 + A_k3;
                                C_mat[j][i] = sum;
                        }
                }
        }
}

/**
 * Verify correctness of parallel implementation by comparing to the sequential version using element wise comparison
 * and a floating point tolerance threshold eps.
 */
bool verify_results(void** sequential, void** parallel, int M, DataType type, double eps)
{
        if (type == DOUBLE_TYPE) {
                double** seq = (double**)sequential;
                double** par = (double**)parallel;
                for (int i = 0; i < M; i++) {
                        for (int j = 0; j < M; j++) {
                                // Check that the difference is acceptable for the given threshold epsilon tolerance
                                if (fabs(seq[i][j] - par[i][j]) > eps) {
                                        return false;
                                }
                        }
                }
        } else {
                float** seq = (float**)sequential;
                float** par = (float**)parallel;
                for (int i = 0; i < M; i++) {
                        for (int j = 0; j < M; j++) {
                                if (fabsf(seq[i][j] - par[i][j]) > eps) {
                                        return false;
                                }
                        }
                }
        }

        // All elements match within the tolerance level so the result is verified
        return true;
}

/**
 * Main driver for the distributed pairwise dot product computation - handles both single process and multiproces modes.
 * The user enters args M cols and N rows to compute the column vector pairwise computation, however these are swapped
 * so A = M rows x N cols as the allocation startegy is row major. This is possible as Amxn row pairwise = Anm col
 * pairwise -> Both M x M result matrices, we just optimise for the row major case.
 */
int main(int argc, char** argv)
{
        // N rows, M cols input -> Implicit transpose as alloc is row major so M rows & N cols
        // rank = current proc id
        // size = number of procs
        int rank, size;
        bool master, multi_proc;
        int M, N; // Stored as M rows, N cols
        DataType type;

        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Determine what the current execution state/flow should be
        master = rank == 0;
        multi_proc = size > 1;

        int validation = validate_input_args(argc - 1, argv + 1, &M, &N, &type);
        if (validation != EXIT_SUCCESS) {
                MPI_Finalize();
                exit(EXIT_FAILURE);
        }

        size_t elem_size = (type == FLOAT_TYPE) ? sizeof(float) : sizeof(double);
        MPI_Datatype mpi_type = (type == FLOAT_TYPE) ? MPI_FLOAT : MPI_DOUBLE;

        // Single process mode which only computes the sequential dot prodcut and does not run the parallel
        // implementation.
        if (!multi_proc) {
                if (master) {
                        // Allocate mats: A input switched M rows x N cols and result C = M x M
                        void** A = alloc_matrix(M, N, elem_size, elem_size);
                        void** C = alloc_matrix(M, M, elem_size, elem_size);
                        init_matrix(A, M, N, type);

                        printf("Performing sequential computation on A: %d rows, %d cols\n", M, N);

                        // Time the sequential version
                        double seq_start = MPI_Wtime();
                        sequential_dotprod(A, C, M, N, type);
                        double seq_end = MPI_Wtime();

                        printf("Sequential computation time: %lf seconds\n", seq_end - seq_start);
                        printf("Peak memory usage: %.6f MB\n", get_peak_memory());

                        if (M <= 16 && N <= 16) {
                                printf("\nInput matrix A (%dx%d):\n", M, N);
                                print_matrix(A, M, N, type);

                                printf("Result matrix C (%dx%d):\n", M, M);
                                print_matrix(C, M, M, type);
                        }

                        free(A);
                        free(C);
                }
        } else {
                // Multi-process mode - run parallel algorithm
                // Calculate work distribution for this process - Each process gets a portion of matrix rows to own and
                // proces
                int local_block, local_offset;
                calculate_block_size(M, size, rank, &local_block, &local_offset);

                // Stores A in row major -> A^T row major = A col major
                // A_loc is the current proc portion of the input matrix
                void** A = NULL;
                void** A_loc = alloc_matrix(local_block, N, elem_size, elem_size);

                // Result matrices
                // C_loc is the current proc portion responsible for in the result matrix
                void** C = NULL;
                void** C_loc = alloc_matrix(local_block, M, elem_size, elem_size);

                // params for scatter and gather -> how much data gets sent and where does it go
                int* send_counts = NULL;
                int* displs = NULL;

                if (master) {
                        // Allocate global matrix -> M vectors/sequences x N elements
                        A = alloc_matrix(M, N, elem_size, elem_size);
                        C = alloc_matrix(M, M, elem_size, elem_size);
                        void** T = alloc_matrix(M, M, elem_size, elem_size);
                        init_matrix(A, M, N, type);

                        printf("Performing sequential computation on A: %d rows, %d cols\n", M, N);

                        double seq_start = MPI_Wtime();
                        sequential_dotprod(A, T, M, N, type);
                        double seq_end = MPI_Wtime();

                        printf("Sequential computation time: %lf seconds\n", seq_end - seq_start);

                        // Scatterv params as each proc will get different amount of data if M % p != 0
                        send_counts = malloc(size * sizeof(int));
                        displs = malloc(size * sizeof(int));

                        for (int i = 0; i < size; i++) {
                                int block, offset;
                                calculate_block_size(M, size, i, &block, &offset);
                                send_counts[i] = N * block; // Every block has N elems per vector
                                displs[i] = N * offset; // start pos
                        }
                        free(T);
                }

                // Use MPI_Scatterv to distribute the input data to all processes - handles the uneven case inseatd of
                // just MPI_Scatter which is fixed
                void* send_data = master ? A[0] : NULL;
                void* recv_data = A_loc[0];
                MPI_Scatterv(send_data, send_counts, displs, mpi_type, recv_data, N * local_block, mpi_type, 0,
                             MPI_COMM_WORLD);
                
                printf("Performing parallel computation on A: %d rows, %d cols\n", M, N);

                // Execute the parallel algorithm and start wall time
                double start_time = MPI_Wtime();
                ring_pairwise(A_loc, C_loc, N, M, rank, size, mpi_type, elem_size, local_block);
                double end_time = MPI_Wtime();

                // Get results
                int* recv_counts = NULL;
                int* recv_displs = NULL;

                if (master) {
                        recv_counts = malloc(size * sizeof(int));
                        recv_displs = malloc(size * sizeof(int));

                        int total_offset = 0;
                        for (int i = 0; i < size; i++) {
                                int block, offset;
                                calculate_block_size(M, size, i, &block, &offset);
                                recv_counts[i] = block * M; // Each process contributes block x M elements
                                recv_displs[i] = total_offset;
                                total_offset += block * M;
                        }
                }

                // Gather the distrib results back to master
                void* gather_send_data = C_loc[0];
                void* gather_recv_data = master ? C[0] : NULL;
                MPI_Gatherv(gather_send_data, local_block * M, mpi_type, gather_recv_data, recv_counts, recv_displs,
                            mpi_type, 0, MPI_COMM_WORLD);

                // Check correctness and report the results
                if (master) {
                        printf("Parallel computation time: %lf seconds\n", end_time - start_time);
                        printf("Peak memory usage: %.6f MB\n", get_peak_memory());

                        // Verification for parallel results
                        void** T = alloc_matrix(M, M, elem_size, elem_size);
                        sequential_dotprod(A, T, M, N, type);

                        double epsilon = (type == FLOAT_TYPE) ? 1e-6 : 1e-8;
                        bool verified = verify_results(T, C, M, type, epsilon);

                        printf("Verification: %s\n", verified ? "PASSED" : "FAILED");

                        // Print matrices for small sizes
                        if (M <= 16 && N <= 16) {
                                printf("\nInput matrix A (%dx%d):\n", M, N);
                                print_matrix(A, M, N, type);

                                printf("Result matrix C (%dx%d):\n", M, M);
                                print_matrix(C, M, M, type);
                        }

                        free(A);
                        free(C);
                        free(T);
                        free(send_counts);
                        free(displs);
                        free(recv_counts);
                        free(recv_displs);
                }

                free(A_loc);
                free(C_loc);
        }

        MPI_Finalize();
        return 0;
}
