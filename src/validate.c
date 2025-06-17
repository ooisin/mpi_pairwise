#include "validate.h"

int validate_input_args(int count, char** input_args, int* cols, int* rows, DataType* type)
{
        if (count != 3) {
                fprintf(stderr, "Invalid number of arguments: Expected 3, but received %d\n", count);
                return EXIT_FAILURE;
        }

        char* endptr;

        long cols_val = strtol(input_args[0], &endptr, 10);
        if (cols_val < 1 || *endptr != '\0') {
                fprintf(stderr,
                        "Invalid dimension argument: Expected columns to be a positive integer, but received %s\n",
                        input_args[0]);
                return EXIT_FAILURE;
        }

        long rows_val = strtol(input_args[1], &endptr, 10);
        if (rows_val < 1 || *endptr != '\0') {
                fprintf(stderr, "Invalid dimension argument: Expected rows to be a positive integer, but received %s\n",
                        input_args[1]);
                return EXIT_FAILURE;
        }

        if (strcmp(input_args[2], "float") == 0) {
                *type = FLOAT_TYPE;
        } else if (strcmp(input_args[2], "double") == 0) {
                *type = DOUBLE_TYPE;
        } else {
                fprintf(stderr, "Invalid datatype argument: Expected \"float\" or \"double\", but received \"%s\"\n",
                        input_args[2]);
                return EXIT_FAILURE;
        }

        *cols = (int)cols_val;
        *rows = (int)rows_val;

        return EXIT_SUCCESS;
}

const char* type_to_str(DataType type)
{
        switch (type) {
        case FLOAT_TYPE:
                return "float";
        case DOUBLE_TYPE:
                return "double";
        default:
                return "invalid";
        }
}
