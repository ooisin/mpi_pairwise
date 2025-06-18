#include "matrix.h"
#include "validate.h"
#include <stdio.h>

static size_t align_up(size_t value, size_t alignment)
{
        return ((value + alignment - 1) / alignment) * alignment;
}

/* The matrix memory allocation follows the efficient 2D array allocation method
 * described by Paul Lucas, 2023, Article available:
 * https://medium.com/@pauljlucas/dynamically-allocating-2d-arrays-efficiently-and-correctly-in-c-1b35384e87bf
 */
void** alloc_matrix(size_t rows, size_t cols, size_t type_size, size_t type_align)
{
        size_t header_size = align_up(sizeof(void*) * rows, type_align);
        size_t row_size = type_size * cols;
        void** row_ptrs = calloc(1, header_size + row_size * rows);
        if (!row_ptrs) {
                return NULL;
        }

        char* mat_data = (char*)row_ptrs + header_size;
        for (size_t row = 0; row < rows; row++) {
                row_ptrs[row] = mat_data + row * row_size;
        }

        return row_ptrs;
}

void init_matrix(void** mat, size_t rows, size_t cols, DataType type)
{
        srand(9730);
        if (type == FLOAT_TYPE) {
                for (size_t i = 0; i < rows; i++) {
                        float* row = (float*)mat[i];
                        for (size_t j = 0; j < cols; j++) {
                                row[j] = (float)rand() / RAND_MAX;
                        }
                }
        } else if (type == DOUBLE_TYPE) {
                for (size_t i = 0; i < rows; i++) {
                        double* row = (double*)mat[i];
                        for (size_t j = 0; j < cols; j++) {
                                row[j] = (double)rand() / RAND_MAX;
                        }
                }
        }

        return;
}

void init_custom_matrix(void** mat, size_t rows, size_t cols, DataType type)
{
        if (type == FLOAT_TYPE) {
                float** matrix = (float**)mat;
                float value = 1.0f;
                for (size_t i = 0; i < rows; i++) {
                        for (size_t j = 0; j < cols; j++) {
                                matrix[i][j] = value++;
                        }
                }
        } else if (type == DOUBLE_TYPE) {
                double** matrix = (double**)mat;
                double value = 1.0;
                for (size_t i = 0; i < rows; i++) {
                        for (size_t j = 0; j < cols; j++) {
                                matrix[i][j] = value++;
                        }
                }
        }
}

void print_matrix(void** mat, size_t rows, size_t cols, DataType type)
{
        for (size_t i = 0; i < rows; i++) {
                if (type == FLOAT_TYPE) {
                        float* row = mat[i];
                        for (size_t j = 0; j < cols; j++) {
                                printf("%-7.2f ", row[j]);
                        }

                } else {
                        double* row = mat[i];
                        for (size_t j = 0; j < cols; j++) {
                                printf("%-7.2lf ", row[j]);
                        }
                }
                printf("\n");
        }
        printf("\n");
}
