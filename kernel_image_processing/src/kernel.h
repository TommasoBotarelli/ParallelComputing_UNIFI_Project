#ifndef KERNEL_H
#define KERNEL_H

struct Kernel {
    int rows;
    int cols;
    int paddingSize;
    float* values;
};


Kernel* initKernel(int rows, int cols, int paddingSize, float* values) {
    Kernel* kernel = new Kernel();
    kernel->rows = rows;
    kernel->cols = cols;
    kernel->paddingSize = paddingSize;
    kernel->values = new float[rows * cols];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            kernel->values[i * cols + j] = values[i * cols + j];
        }
    }
    return kernel;
}

#endif