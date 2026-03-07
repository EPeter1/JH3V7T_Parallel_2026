#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

const int SAMPLE_SIZE = 1000;

int generate_number(int lower, int upper);

int main(void)
{
    int i;
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
    char* kernel_code = read_kernel("kernels/rank.cl");

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_code, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Build error! Code: %d\n", err);
        return 0;
    }
    cl_kernel kernel = clCreateKernel(program, "get_rank", NULL);

    // Create the host buffer and initialize it
    int* host_buffer_input = (int*)malloc(SAMPLE_SIZE * sizeof(int));
    int* host_buffer_rank = (int*)malloc(SAMPLE_SIZE * sizeof(int));

    srand(time(0));
    for (i = 0; i < SAMPLE_SIZE; ++i) {
        host_buffer_input[i] = generate_number(1, 100);
    }

    // Create the device buffer
    cl_mem device_buffer_input = clCreateBuffer(context, CL_MEM_READ_ONLY, SAMPLE_SIZE * sizeof(int), NULL, NULL);
    cl_mem device_buffer_rank = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SAMPLE_SIZE * sizeof(int), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_buffer_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_buffer_rank);
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&SAMPLE_SIZE);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, NULL);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(
        command_queue,
        device_buffer_input,
        CL_TRUE,
        0,
        SAMPLE_SIZE * sizeof(int),
        host_buffer_input,
        0,
        NULL,
        NULL
    );

    // Size specification
    size_t local_work_size = 256;
    size_t n_work_groups = (SAMPLE_SIZE + local_work_size - 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;

    // Apply the kernel on the range
    clEnqueueNDRangeKernel(
        command_queue,
        kernel,
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
        device_buffer_rank,
        CL_TRUE,
        0,
        SAMPLE_SIZE * sizeof(int),
        host_buffer_rank,
        0,
        NULL,
        NULL
    );

    for (i = 0; i < SAMPLE_SIZE; ++i) {
        printf("%d, ", host_buffer_rank[i]);
    }

    // Release the resources
    clReleaseMemObject(device_buffer_input);
    clReleaseMemObject(device_buffer_rank);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(host_buffer_input);
    free(host_buffer_rank);
}

int generate_number(int lower, int upper) {

    return rand() % (upper - lower + 1) + lower;
}
