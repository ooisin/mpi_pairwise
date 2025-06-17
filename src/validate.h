#ifndef VALIDATE_H
#define VALIDATE_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { FLOAT_TYPE, DOUBLE_TYPE } DataType;
// TODO: Cite
int validate_input_args(int count, char** input_args, int* cols, int* rows, DataType* type);
const char* type_to_str(DataType type);

#endif // VALIDATE_H
