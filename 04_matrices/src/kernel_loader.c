#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>

char* read_kernel(const char* file_path)
{
    FILE* file = fopen(file_path, "rb");

    if (!file) {
        perror("Could not open file for reading!");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long int file_size = ftell(file);
    rewind(file);

    char* buffer = (char*)malloc(file_size + 1);

    if (!buffer) {
        perror("Memory allocation failed!");
        fclose(file);
        
        return NULL;
    }

    fread(buffer, sizeof(char), file_size, file);
    buffer[file_size] = '\0';

    fclose(file);
    return buffer;
}
