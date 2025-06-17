#ifndef MATRIX_H
#define MATRIX_H

#include "validate.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void** alloc_matrix(size_t rows, size_t cols, size_t type_size, size_t type_align);
void init_matrix(void** mat, size_t rows, size_t cols, DataType type);
void init_custom_matrix(void** mat, size_t rows, size_t cols, DataType type);
void print_matrix(void** mat, size_t rows, size_t cols, DataType type);

#endif // MATRIX_H
