#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

int generate_number(int lower, int upper);
void print_matrix(int* matrix, int rows, int columns);
void print_vector(int* vector, int size);

int main(void)
{
    cl_int err;

    // Get platform
    cl_uint n_platforms;
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
        return 0;
    }

    // Get device
    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(
        platform_id,
        CL_DEVICE_TYPE_GPU,
        1,
        &device_id,
        &n_devices
    );
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
        return 0;
    }

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

    // Build the program
    char* kernel_code = read_kernel("kernels/matrices.cl");

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_code, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Build error! Code: %d\n", err);
        return 0;
    }

    cl_kernel kernel_transpose = clCreateKernel(program, "transpose_matrix", NULL);
    cl_kernel kernel_product = clCreateKernel(program, "multiply_matrices", NULL);
    cl_kernel kernel_sum_rows = clCreateKernel(program, "sum_rows_matrix", NULL);
    cl_kernel kernel_sum_columns = clCreateKernel(program, "sum_columns_matrix", NULL);

    // Create the host buffer and initialize it
    int rows = 5;
    int columns = 5;
    int element_count = rows * columns;

    int* host_buffer_matrix_A = (int*)malloc(element_count * sizeof(int));
    int* host_buffer_matrix_B = (int*)malloc(element_count * sizeof(int));
    int* host_buffer_transpose = (int*)malloc(element_count * sizeof(int));
    int* host_buffer_product = (int*)malloc(element_count * sizeof(int));
    int* host_buffer_sum_rows = (int*)malloc(rows * sizeof(int));
    int* host_buffer_sum_columns = (int*)malloc(columns * sizeof(int));

    srand(time(0));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            int index = i * columns + j;
            host_buffer_matrix_A[index] = generate_number(-10, 10);
            host_buffer_matrix_B[index] = generate_number(-10, 10);
        }
    }

    // Create the device buffer
    cl_mem device_buffer_matrix_A = clCreateBuffer(context, CL_MEM_READ_ONLY, element_count * sizeof(int), NULL, NULL);
    cl_mem device_buffer_matrix_B = clCreateBuffer(context, CL_MEM_READ_ONLY, element_count * sizeof(int), NULL, NULL);
    cl_mem device_buffer_transpose = clCreateBuffer(context, CL_MEM_WRITE_ONLY, element_count * sizeof(int), NULL, NULL);
    cl_mem device_buffer_product = clCreateBuffer(context, CL_MEM_WRITE_ONLY, element_count * sizeof(int), NULL, NULL);
    cl_mem device_buffer_sum_rows = clCreateBuffer(context, CL_MEM_WRITE_ONLY, rows * sizeof(int), NULL, NULL);
    cl_mem device_buffer_sum_columns = clCreateBuffer(context, CL_MEM_WRITE_ONLY, columns * sizeof(int), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel_transpose, 0, sizeof(cl_mem), (void*)&device_buffer_matrix_A);
    clSetKernelArg(kernel_transpose, 1, sizeof(cl_mem), (void*)&device_buffer_transpose);
    clSetKernelArg(kernel_transpose, 2, sizeof(int), (void*)&rows);
    clSetKernelArg(kernel_transpose, 3, sizeof(int), (void*)&columns);

    clSetKernelArg(kernel_product, 0, sizeof(cl_mem), (void*)&device_buffer_matrix_A);
    clSetKernelArg(kernel_product, 1, sizeof(cl_mem), (void*)&device_buffer_matrix_B);
    clSetKernelArg(kernel_product, 2, sizeof(cl_mem), (void*)&device_buffer_product);
    clSetKernelArg(kernel_product, 3, sizeof(int), (void*)&rows);
    clSetKernelArg(kernel_product, 4, sizeof(int), (void*)&columns);
    clSetKernelArg(kernel_product, 5, sizeof(int), (void*)&columns);

    clSetKernelArg(kernel_sum_rows, 0, sizeof(cl_mem), (void*)&device_buffer_matrix_A);
    clSetKernelArg(kernel_sum_rows, 1, sizeof(cl_mem), (void*)&device_buffer_sum_rows);
    clSetKernelArg(kernel_sum_rows, 2, sizeof(int), (void*)&rows);
    clSetKernelArg(kernel_sum_rows, 3, sizeof(int), (void*)&columns);

    clSetKernelArg(kernel_sum_columns, 0, sizeof(cl_mem), (void*)&device_buffer_matrix_A);
    clSetKernelArg(kernel_sum_columns, 1, sizeof(cl_mem), (void*)&device_buffer_sum_columns);
    clSetKernelArg(kernel_sum_columns, 2, sizeof(int), (void*)&rows);
    clSetKernelArg(kernel_sum_columns, 3, sizeof(int), (void*)&columns);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, NULL);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(
        command_queue,
        device_buffer_matrix_A,
        CL_TRUE,
        0,
        element_count * sizeof(int),
        host_buffer_matrix_A,
        0,
        NULL,
        NULL
    );

    clEnqueueWriteBuffer(
        command_queue,
        device_buffer_matrix_B,
        CL_TRUE,
        0,
        element_count * sizeof(int),
        host_buffer_matrix_B,
        0,
        NULL,
        NULL
    );

    // Size specification
    size_t local_work_size = 32;
    size_t n_work_groups = (rows + local_work_size - 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;

    size_t local_work_size_2D[2] = {16, 16};
    size_t global_work_size_2D[2] = {
        ((element_count + local_work_size_2D[0] - 1) / local_work_size_2D[0]) * local_work_size_2D[0],
        ((element_count + local_work_size_2D[1] - 1) / local_work_size_2D[1]) * local_work_size_2D[1]
    };

    // Apply the kernel on the range
    clEnqueueNDRangeKernel(
        command_queue,
        kernel_transpose,
        2,
        NULL,
        global_work_size_2D,
        local_work_size_2D,
        0,
        NULL,
        NULL
    );

    clEnqueueNDRangeKernel(
        command_queue,
        kernel_product,
        2,
        NULL,
        global_work_size_2D,
        local_work_size_2D,
        0,
        NULL,
        NULL
    );

    clEnqueueNDRangeKernel(
        command_queue,
        kernel_sum_rows,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
    );

    clEnqueueNDRangeKernel(
        command_queue,
        kernel_sum_columns,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
    );

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(
        command_queue,
        device_buffer_transpose,
        CL_TRUE,
        0,
        element_count * sizeof(int),
        host_buffer_transpose,
        0,
        NULL,
        NULL
    );

    clEnqueueReadBuffer(
        command_queue,
        device_buffer_product,
        CL_TRUE,
        0,
        element_count * sizeof(int),
        host_buffer_product,
        0,
        NULL,
        NULL
    );

    clEnqueueReadBuffer(
        command_queue,
        device_buffer_sum_rows,
        CL_TRUE,
        0,
        rows * sizeof(int),
        host_buffer_sum_rows,
        0,
        NULL,
        NULL
    );

    clEnqueueReadBuffer(
        command_queue,
        device_buffer_sum_columns,
        CL_TRUE,
        0,
        columns * sizeof(int),
        host_buffer_sum_columns,
        0,
        NULL,
        NULL
    );

    printf("A matrix\n------------------------\n");
    print_matrix(host_buffer_matrix_A, rows, columns);
    printf("\nB matrix\n------------------------\n");
    print_matrix(host_buffer_matrix_B, rows, columns);

    printf("\nA^T\n------------------------\n");
    print_matrix(host_buffer_transpose, rows, columns);
    printf("\nA * B\n------------------------\n");
    print_matrix(host_buffer_product, rows, columns);
    printf("\nRow sums (A)\n------------------------\n");
    print_vector(host_buffer_sum_rows, rows);
    printf("\nColumn sums (A)\n------------------------\n");
    print_vector(host_buffer_sum_columns, columns);

    // Release the resources
    clReleaseMemObject(device_buffer_matrix_A);
    clReleaseMemObject(device_buffer_matrix_B);
    clReleaseMemObject(device_buffer_transpose);
    clReleaseMemObject(device_buffer_product);
    clReleaseMemObject(device_buffer_sum_rows);
    clReleaseMemObject(device_buffer_sum_columns);

    clReleaseKernel(kernel_transpose);
    clReleaseKernel(kernel_product);
    clReleaseKernel(kernel_sum_rows);
    clReleaseKernel(kernel_sum_columns);

    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(host_buffer_matrix_A);
    free(host_buffer_matrix_B);
    free(host_buffer_transpose);
    free(host_buffer_product);
    free(host_buffer_sum_rows);
    free(host_buffer_sum_columns);
}

int generate_number(int lower, int upper) {

    return rand() % (upper - lower + 1) + lower;
}

void print_matrix(int* matrix, int rows, int columns) {

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            printf("%4d ", matrix[i * columns + j]);
        }
        printf("\n");
    }
}

void print_vector(int* vector, int size) {

    for (int i = 0; i < size; i++) {
        printf("%4d ", vector[i]);
    }
    printf("\n");
}
