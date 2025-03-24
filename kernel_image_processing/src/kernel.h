#ifndef KERNEL_H
#define KERNEL_H

#include <cmath>
#include <iostream>

#define IDENTITY 1
#define BLUR 2
#define EDGE 3
#define GAUSSIAN 4

struct Kernel {
    int rows;
    int cols;
    int paddingSize;
    float* values;
    int type;
};


const char* getKernelTypeString(int type) {
    const char* kernelTypeString;

    switch (type)
    {
    case IDENTITY:
        kernelTypeString = "IDENTITY";
        break;
    case BLUR:
        kernelTypeString = "BLUR";
        break;
    case EDGE:
        kernelTypeString = "EDGE";
        break;
    case GAUSSIAN:
        kernelTypeString = "GAUSSIAN";
        break;
    default:
        std::cerr << "Error: Kernel type not supported." << std::endl;
        return nullptr;
    }

    return kernelTypeString;
}


Kernel* initKernel(int size, int paddingSize, const float* values) {
    Kernel* kernel = new Kernel();
    kernel->rows = size;
    kernel->cols = size;
    kernel->paddingSize = paddingSize;
    kernel->values = new float[size * size];
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            kernel->values[i * size + j] = values[i * size + j];
        }
    }
    return kernel;
}

float* getIdentityMatrix(int size) {
    float* values = new float[size*size];

    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++) {
            values[i * size + j] = 0;
            if (i == size/2 && j == size/2) {
                values[i * size + j] = 1;
            }
        }
    }

    return values;
}

float* getBlurMatrix(int size) {
    float* values = new float[size*size];
    float blurValue = 1 / (float)(size * size);

    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++) {
            values[i * size + j] = blurValue;
        }
    }

    return values;
}

float* getEdgeMatrix(int size) {
    float* values = new float[size*size];

    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++) {
            values[i * size + j] = -1;
            if (i == size/2 && j == size/2) {
                values[i * size + j] = size*size - 1;
            }
        }
    }

    return values;
}

float calculateSigma(int size) {
    return 0.3 * (((float)size - 1) * 0.5 - 1) + 0.8;
}

float* getGaussianMatrix(int size) {
    float sigma = calculateSigma(size);
    float* values = new float[size * size];
    float sum = 0.0;
    int halfSize = size / 2;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int x = i - halfSize;
            int y = j - halfSize;
            values[i * size + j] = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            sum += values[i * size + j];
        }
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            values[i * size + j] /= sum;
        }
    }

    return values;
}

Kernel* getKernel(int size, int type) {
    Kernel* kernel = new Kernel();
    kernel->cols = size;
    kernel->rows = size;
    kernel->paddingSize = (int)size/2;

    float* values;

    switch (type)
    {
    case IDENTITY:
        values =  getIdentityMatrix(size);
        break;
    case BLUR:
        values =  getBlurMatrix(size);
        break;
    case EDGE:
        values =  getEdgeMatrix(size);
        break;
    case GAUSSIAN:
        values =  getGaussianMatrix(size);
        break;
    default:
        std::cerr << "Error: Kernel type not supported." << std::endl;
        return nullptr;
    }

    kernel->values = values;
    kernel->type = type;

    return kernel;
}

void printKernel(const Kernel* kernel) {
    std::cout << "Kernel type: " << getKernelTypeString(kernel->type) << std::endl;
    std::cout << "Kernel size: " << kernel->rows << std::endl;
    std::cout << "Kernel padding: " << kernel->paddingSize << std::endl;
    std::cout << "Values:" << std::endl;
    for (int i = 0; i < kernel->rows; i++) {
        for (int j = 0; j < kernel->cols; j++) {
            std::cout << std::fixed << std::setprecision(3) << kernel->values[i * kernel->cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

#endif