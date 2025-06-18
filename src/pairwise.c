#include "matrix.h"
#include "mpi.h"
#include "validate.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/**
 * Determines block size/width (number of columns) and handles the imperfect
 * distribution case M % p != 0 by giving some processes an extra column.
 * The first M % size procs get M/size + 1 cols, the remaining get M/size cols
 * to minimise the load imbalance across all processes fairly. Maintaining the
 * offset ensures no columns overlap aacross blocks.
 */
void calculate_block_size(int M, int size, int rank, int* block_size, int* offset)
{
        // TODO: Check when p = 1
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

void compute_dot_product(void** A_local, void** A_received, void** C_local, int N, int M, int local_block_rows,
                         int recv_block_rows, int rank, int step, int size, MPI_Datatype dtype)
{
        if (dtype == MPI_DOUBLE) {
                double** A_loc = (double**)A_local;
                double** A_rec = (double**)A_received;
                double** C_loc = (double**)C_local;

                // Calculate local offset for this rank
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

                                // 4-way loop unrolling for dot product computation
                                double A_k = 0.0;
                                double A_k1 = 0.0;
                                double A_k2 = 0.0;
                                double A_k3 = 0.0;

                                int k;

                                // Row-based access: A_loc[i][k] gives element k of vector i
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

                                // Store result in appropriate position of Gram matrix
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

void ring_pairwise(void** A_loc, void** C_loc, int N, int M, int rank, int size, MPI_Datatype type, size_t elem_size,
                   int local_block_rows)
{
        // Ring topology neighbours
        int next_rank = (rank + 1) % size;
        int prev_rank = (rank + size - 1) % size;
        int max_block_rows = (M + size - 1) / size; // Max rows any process might have

        // Communication buffers for ring shifts (store complete rows)
        void** send_buf = alloc_matrix(max_block_rows, N, elem_size, elem_size);
        void** recv_buf = alloc_matrix(max_block_rows, N, elem_size, elem_size);

        // Initialize send buffer with local block (copy entire rows)
        for (int i = 0; i < local_block_rows; i++) {
                memcpy(send_buf[i], A_loc[i], N * elem_size);
        }
        int current_block_rows = local_block_rows;

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

                        // Swap buffers for next iteration
                        void** temp = send_buf;
                        send_buf = recv_buf;
                        recv_buf = temp;
                        current_block_rows = next_block_rows;
                }
        }

        free(send_buf);
        free(recv_buf);
}

void sequential_dotprod(void** A, void** C, int rows, int cols, DataType type)
{
        if (type == DOUBLE_TYPE) {
                double** A_mat = (double**)A;
                double** C_mat = (double**)C;
                
                // Compute result mat C 
                for (int i = 0; i < rows; i++) {
                        // Only compute upper triangle
                        for (int j = 0; j <= i; j++) {  
                                double A_k = 0.0;
                                double A_k1 = 0.0;
                                double A_k2 = 0.0;
                                double A_k3 = 0.0;

                                int k;
                                
                                // Dot product of row i,j
                                for (k = 0; k < cols - 3; k += 4){
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
                                double sum = A_k + A_k1 + A_k2 + A_k3;
                                C_mat[i][j] = sum;
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
                                for (k = 0; k < cols - 3; k += 4){
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
                                C_mat[i][j] = sum;
                                C_mat[j][i] = sum;
                        }
                }
        }
}

int main(int argc, char** argv)
{
        // N rows, M cols input
        // Implicit transpose as alloc is row major so M rows & N cols
        int rank, size;
        bool master, multi_proc;
        int M, N;
        DataType type;

        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        master = rank == 0;
        multi_proc = size > 0;

        int validation = validate_input_args(argc - 1, argv + 1, &M, &N, &type);
        if (validation != EXIT_SUCCESS) {
                MPI_Finalize();
                exit(EXIT_FAILURE);
        }

        // Calculate local block size for this rank
        int local_block, local_offset;
        calculate_block_size(M, size, rank, &local_block, &local_offset);

        size_t elem_size = (type == FLOAT_TYPE) ? sizeof(float) : sizeof(double);
        MPI_Datatype mpi_type = (type == FLOAT_TYPE) ? MPI_FLOAT : MPI_DOUBLE;

        // Stores A in row major -> A^T row major = A col major
        void** A = NULL;
        void** A_loc = alloc_matrix(local_block, N, elem_size, elem_size);

        // Result matrices
        void** C = NULL;
        void** C_loc = alloc_matrix(local_block, M, elem_size, elem_size);

        // params for scatter and gather
        int* send_counts = NULL;
        int* displs = NULL;

        if (master) {
                // Allocate global matrix -> M vectors/sequences x N elements
                A = alloc_matrix(M, N, elem_size, elem_size);
                C = alloc_matrix(M, M, elem_size, elem_size);
                void** T = alloc_matrix(M, M, elem_size, elem_size);
                init_matrix(A, M, N, type);

                printf("M=%d, N=%d\n", M, N);
                printf("Matrix A - vectors as rows:\n");
                //print_matrix(A, M, N, type);
                

                double seq_start = MPI_Wtime();
                sequential_dotprod(A, T, M, N, type);
                double seq_end = MPI_Wtime();
                printf("Sequential computation time: %f seconds\n", seq_end - seq_start);
                //print_matrix(T, M, M, type);

                // Scatterv params
                send_counts = malloc(size * sizeof(int));
                displs = malloc(size * sizeof(int));

                for (int i = 0; i < size; i++) {
                        int block, offset;
                        calculate_block_size(M, size, i, &block, &offset);
                        send_counts[i] = N * block; // Every block has N elems per vector
                        displs[i] = N * offset;
                }
        }

        // Use MPI_Scatterv for uneven distribution case
        void* send_data = master ? A[0] : NULL;
        void* recv_data = A_loc[0];
        MPI_Scatterv(send_data, send_counts, displs, mpi_type, recv_data, N * local_block, mpi_type, 0, MPI_COMM_WORLD);

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

        void* gather_send_data = C_loc[0];
        void* gather_recv_data = master ? C[0] : NULL;
        MPI_Gatherv(gather_send_data, local_block * M, mpi_type, gather_recv_data, recv_counts, recv_displs, mpi_type, 0, MPI_COMM_WORLD);

        if (master) {
                printf("Computation time: %f seconds\n", end_time - start_time);
                //printf("Result matrix C:\n");
                print_matrix(C, M, M, type);

                free(A);
                free(C);
                free(send_counts);
                free(displs);
                free(recv_counts);
                free(recv_displs);
        }

        free(A_loc);
        free(C_loc);
        MPI_Finalize();
        return 0;
}
