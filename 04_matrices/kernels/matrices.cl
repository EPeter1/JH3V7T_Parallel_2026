__kernel void transpose_matrix(
    __global const int* matrix,
    __global int* transpose,
    const int rows,
    const int columns)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= rows || j >= columns) {
        return;
    }

    transpose[j * rows + i] = matrix[i * columns + j];
}

__kernel void multiply_matrices(
    __global const int* matrix_A,
    __global const int* matrix_B,
    __global int* product,
    const int rows_A,
    const int columns_A,
    const int columns_B)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= rows_A || j >= columns_B) {
        return;
    }

    int sum = 0;

    for (int k = 0; k < columns_A; k++) {
        sum += matrix_A[i * columns_A + k] * matrix_B[k * columns_B + j];
    }

    product[i * columns_B + j] = sum;
}

__kernel void sum_rows_matrix(
    __global const int* matrix,
    __global int* sum,
    const int rows,
    const int columns)
{
    int i = get_global_id(0);

    if (i >= rows) {
        return;
    }

    int sum_rows = 0;

    for (int j = 0; j < columns; j++) {
        sum_rows += matrix[i * columns + j];
    }

    sum[i] = sum_rows;
}

__kernel void sum_columns_matrix(
    __global const int* matrix,
    __global int* sum,
    const int rows,
    const int columns)
{
    int i = get_global_id(0);

    if (i >= columns) {
        return;
    }

    int sum_columns = 0;

    for (int j = 0; j < rows; j++) {
        sum_columns += matrix[j * columns + i];
    }

    sum[i] = sum_columns;
}
