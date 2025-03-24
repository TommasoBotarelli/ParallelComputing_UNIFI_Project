#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/opencv.hpp>
#include "kernel.h"
#include "cuda.h"
#include <algorithm>

__constant__ float d_kernel_const[4096];
__constant__ int const_resultCols;
__constant__ int const_resultRows;
__constant__ int const_paddingSize;
__constant__ int const_inputImageCols;
__constant__ int const_kernelCols;
__constant__ int const_N;
__constant__ int const_channelN;
__constant__ int const_channelNInput;

struct Image {
    long long int rows;
    long long int cols;

    short int* channels[3];
};


Image* initColoredImage(int rows, int cols) {
    Image* image = new Image();
    image->rows = rows;
    image->cols = cols;
    image->channels[0] = new short int[rows * cols];
    image->channels[1] = new short int[rows * cols];
    image->channels[2] = new short int[rows * cols];
    return image;
}


Image* getTestImage(int rows, int cols, int numRows = 2, int numCols=2) {
    Image* image = initColoredImage(rows, cols);

    int cellHeight = rows / numRows;
    int cellWidth = cols / numCols;

    for (int channelIndex = 0; channelIndex < 3; channelIndex++){
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                int rowIndex = i / cellHeight;
                int colIndex = j / cellWidth;

                if ((rowIndex + colIndex) % 2 == 0) {
                    image->channels[channelIndex][i * cols + j] = 0;
                } else {
                    image->channels[channelIndex][i * cols + j] = 255;
                }
            }
        }
    }

    return image;
}

Image* getBorderedImage(Image* image, int paddingSize) {
    int resultNumRows = image->rows + 2*paddingSize;
    int resultNumCols = image->cols + 2*paddingSize;

    Image* resultImage = initColoredImage(resultNumRows, resultNumCols);

    for (int i = 0; i < resultNumRows; i++) {
        for (int j = 0; j < resultNumCols; j++) {
            if (i < paddingSize || i >= image->rows + paddingSize || j < paddingSize || j >= image->cols + paddingSize) {
                resultImage->channels[0][i * resultNumCols + j] = 0;
                resultImage->channels[1][i * resultNumCols + j] = 0;
                resultImage->channels[2][i * resultNumCols + j] = 0;
            }
            else {
                resultImage->channels[0][i * resultNumCols + j] = image->channels[0][(i-paddingSize) * image->cols + (j-paddingSize)];
                resultImage->channels[1][i * resultNumCols + j] = image->channels[1][(i-paddingSize) * image->cols + (j-paddingSize)];
                resultImage->channels[2][i * resultNumCols + j] = image->channels[2][(i-paddingSize) * image->cols + (j-paddingSize)];
            }
        }
    }

    return resultImage;
}


Image* initColoredImage(cv::Mat* inputImage) {
    Image* image = initColoredImage(inputImage->rows, inputImage->cols);

    for (int i = 0; i < inputImage->rows; ++i) {
        for (int j = 0; j < inputImage->cols; ++j) {
            image->channels[0][i * inputImage->cols + j] = static_cast<int>(inputImage->at<cv::Vec3b>(i, j)[0]);
            image->channels[1][i * inputImage->cols + j] = static_cast<int>(inputImage->at<cv::Vec3b>(i, j)[1]);
            image->channels[2][i * inputImage->cols + j] = static_cast<int>(inputImage->at<cv::Vec3b>(i, j)[2]);
        }
    }

    return image;
}


short int* linearizeImage(Image* image) {
    long long int channelSize = image->rows * image->cols;
    long long int N = channelSize * 3;
    short int* pixels = new short int[N];

    for (long long int channelIndex = 0; channelIndex < 3; channelIndex++) {
        for (long long int i = 0; i < image->rows; i++) {
            for (long long int j = 0; j < image->cols; j++) {
                pixels[channelIndex*channelSize + i*(long long int)image->cols + j] = image->channels[channelIndex][i*image->cols + j];
            }
        }
    }

    return pixels;
}


short int* linearizeImage_parallel(Image* image) {
    long long int channelSize = image->rows * image->cols;
    long long int N = channelSize * 3;
    short int* pixels = new short int[N];

    #pragma omp parallel for collapse(2)
    for (long long int channelIndex = 0; channelIndex < 3; channelIndex++) {
        for (long long int i = 0; i < image->rows; i++) {
            for (long long int j = 0; j < image->cols; j++) {
                pixels[channelIndex*channelSize + i*(long long int)image->cols + j] = image->channels[channelIndex][i*image->cols + j];
            }
        }
    }

    return pixels;
}


void delinearizeImage(short int* pixels, Image* image) {
    long long int channelSize = image->rows * image->cols;

    for (long long int channelIndex = 0; channelIndex < 3; channelIndex++) {
        for (long long int i = 0; i < image->rows; i++) {
            for (long int j = 0; j < image->cols; j++) {
                image->channels[channelIndex][i*image->cols + j] = pixels[channelIndex*channelSize + i*image->cols + j];
            }
        }
    }
}


void delinearizeImage_parallel(short int* pixels, Image* image) {
    long long int channelSize = image->rows * image->cols;

    #pragma omp parallel for collapse(2)
    for (long long int channelIndex = 0; channelIndex < 3; channelIndex++) {
        for (long long int i = 0; i < image->rows; i++) {
            for (long int j = 0; j < image->cols; j++) {
                image->channels[channelIndex][i*image->cols + j] = pixels[channelIndex*channelSize + i*image->cols + j];
            }
        }
    }
}


Image* processImage(Image* inputImage, Kernel* kernel) {
    int newRows = inputImage->rows - 2 * kernel->paddingSize;
    int newCols = inputImage->cols - 2 * kernel->paddingSize;
    Image* processedImage = initColoredImage(newRows, newCols);

    int kernelSize = 2 * kernel->paddingSize + 1;
    float* kernelValues = kernel->values;

    for (int channelIndex = 0; channelIndex < 3; channelIndex++) {
        short int* inputChannel = inputImage->channels[channelIndex];
        short int* outputChannel = processedImage->channels[channelIndex];

        for (int i = kernel->paddingSize; i < inputImage->rows - kernel->paddingSize; ++i) {
            for (int j = kernel->paddingSize; j < inputImage->cols - kernel->paddingSize; ++j) {
                float sum = 0.0f;
                int outIndex = (i - kernel->paddingSize) * newCols + (j - kernel->paddingSize);

                for (int k = 0; k < kernelSize; ++k) {
                    for (int l = 0; l < kernelSize; ++l) {
                        int imgIndex = (i + k - kernel->paddingSize) * inputImage->cols + (j + l - kernel->paddingSize);
                        int kernelIndex = k * kernelSize + l;
                        sum += kernelValues[kernelIndex] * inputChannel[imgIndex];
                    }
                }

                outputChannel[outIndex] = std::max(0.0f, std::min(255.0f, sum));
            }
        }
    }

    return processedImage;
}


__global__ void kernelProcessing_oneChannel(short int* inputImage, 
                                 float* kernel, 
                                 short int* resultImage, 
                                 int resultCols, 
                                 int paddingSize,
                                 int inputImageCols,
                                 int kernelCols,
                                 int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = index / resultCols;
    int j = index % resultCols;

    if (index < N){
        float sum = 0.0f;
        for (int k = -paddingSize; k <= paddingSize; ++k) {
            for (int l = -paddingSize; l <= paddingSize; ++l) {
                int kernelIndex = (k+paddingSize) * kernelCols + (l+paddingSize);
                int imageIndex =  (i+paddingSize+k) * inputImageCols + (j+paddingSize+l);
                sum += kernel[kernelIndex] * inputImage[imageIndex];
            }
        }
        if (sum < 0) {
            sum = 0;
        }
        if (sum > 255) {
            sum = 255;
        }
        resultImage[i * resultCols + j] = sum;
    }
}



Image* processImage_CUDA_oneChannel(Image* h_image, Kernel* h_kernel) {
    Image* h_resultImage = initColoredImage(h_image->rows - 2 * h_kernel->paddingSize, h_image->cols - 2 * h_kernel->paddingSize);

    int numThreads = h_resultImage->rows * h_resultImage->cols;
    int threadsPerBlock = 1024;
    int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
    
    short int* d_resultImage;
    short int* d_image;
    float* d_kernel;

    cudaMalloc(&d_resultImage, (size_t)h_resultImage->rows*h_resultImage->cols*sizeof(short int));
    cudaMalloc(&d_image, (size_t)h_image->rows*h_image->cols*sizeof(short int));
    cudaMalloc(&d_kernel, (size_t)h_kernel->rows*h_kernel->cols*sizeof(float));

    cudaMemcpy(d_kernel, h_kernel->values, h_kernel->rows*h_kernel->cols*sizeof(float), cudaMemcpyHostToDevice);
    
    for (int channelIndex = 0; channelIndex < 3; channelIndex++){
        cudaMemcpy(d_image, h_image->channels[channelIndex], h_image->rows*h_image->cols*sizeof(short int), cudaMemcpyHostToDevice);
        kernelProcessing_oneChannel<<<numBlocks, threadsPerBlock>>>(d_image, d_kernel, d_resultImage, h_resultImage->cols, h_kernel->paddingSize, h_image->cols, h_kernel->cols, h_resultImage->rows*h_resultImage->cols);
        cudaMemcpy(h_resultImage->channels[channelIndex], d_resultImage, h_resultImage->cols*h_resultImage->rows*sizeof(short int), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_image);
    cudaFree(d_resultImage);
    cudaFree(d_kernel);

    return h_resultImage;
}

__global__ void kernelProcessing_threeChannel_constant(short int* inputImage, short int* resultImage) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int channelIndex = blockIdx.z;

    if (i < const_resultRows && j < const_resultCols){
        float sum = 0.0f;
        for (int k = -const_paddingSize; k <= const_paddingSize; ++k) {
            for (int l = -const_paddingSize; l <= const_paddingSize; ++l) {
                int kernelIndex = (k+const_paddingSize) * const_kernelCols + (l+const_paddingSize);
                int imageIndex = channelIndex * const_channelNInput + (i+const_paddingSize+k) * const_inputImageCols + (j+const_paddingSize+l);
                sum += d_kernel_const[kernelIndex] * inputImage[imageIndex];
            }
        }
        if (sum < 0) {
            sum = 0;
        }
        if (sum > 255) {
            sum = 255;
        }
        resultImage[channelIndex * const_channelN + i * const_resultCols + j] = sum;
    }
}


Image* processImage_CUDA_threeChannel_constant(Image* h_image, Kernel* h_kernel) {
    Image* h_resultImage = initColoredImage(h_image->rows - 2 * h_kernel->paddingSize, h_image->cols - 2 * h_kernel->paddingSize);
    long long int totalResultPixels = h_resultImage->rows * h_resultImage->cols * (long long int)3;
    
    short int* h_resultImagePixels = new short int[totalResultPixels];
    short int* h_imagePixels = linearizeImage(h_image);

    dim3 threadsPerBlock(32, 32); 
    dim3 numBlocks((h_resultImage->rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (h_resultImage->cols + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 3);

    short int* d_resultImagePixels;
    short int* d_imagePixels;

    long long int totalInputPixels = h_image->rows * h_image->cols * 3;

    cudaMalloc(&d_resultImagePixels, (size_t)totalResultPixels*sizeof(short int));
    cudaMalloc(&d_imagePixels, (size_t)totalInputPixels*sizeof(short int));

    cudaMemcpyToSymbol(d_kernel_const, h_kernel->values, h_kernel->rows * h_kernel->cols * sizeof(float));
    cudaMemcpyToSymbol(const_resultCols, &h_resultImage->cols, sizeof(int));
    cudaMemcpyToSymbol(const_resultRows, &h_resultImage->rows, sizeof(int));
    cudaMemcpyToSymbol(const_paddingSize, &h_kernel->paddingSize, sizeof(int));
    cudaMemcpyToSymbol(const_inputImageCols, &h_image->cols, sizeof(int));
    cudaMemcpyToSymbol(const_kernelCols, &h_kernel->cols, sizeof(int));
    cudaMemcpyToSymbol(const_N, &totalResultPixels, sizeof(int));
    int channelN = h_resultImage->rows * h_resultImage->cols;
    cudaMemcpyToSymbol(const_channelN, &channelN, sizeof(int));
    int channelNInput = h_image->rows * h_image->cols;
    cudaMemcpyToSymbol(const_channelNInput, &channelNInput, sizeof(int));

    cudaMemcpy(d_imagePixels, h_imagePixels, totalInputPixels*sizeof(short int), cudaMemcpyHostToDevice);

    kernelProcessing_threeChannel_constant<<<numBlocks, threadsPerBlock>>>(d_imagePixels, d_resultImagePixels);
    
    cudaMemcpy(h_resultImagePixels, d_resultImagePixels, totalResultPixels*sizeof(short int), cudaMemcpyDeviceToHost);

    delinearizeImage(h_resultImagePixels, h_resultImage);

    delete[] h_resultImagePixels;
    delete[] h_imagePixels;

    cudaFree(d_imagePixels);
    cudaFree(d_resultImagePixels);

    return h_resultImage;
}


__global__ void kernelProcessing_constant_noGrid(short int* inputImage, short int* resultImage) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = index / const_resultCols;
    int j = index % const_resultCols;
    int channelIndex = blockIdx.z;

    if (index < const_N){
        float sum = 0.0f;
        for (int k = -const_paddingSize; k <= const_paddingSize; ++k) {
            for (int l = -const_paddingSize; l <= const_paddingSize; ++l) {
                int kernelIndex = (k+const_paddingSize) * const_kernelCols + (l+const_paddingSize);
                int imageIndex = channelIndex * const_channelNInput + (i+const_paddingSize+k) * const_inputImageCols + (j+const_paddingSize+l);
                sum += d_kernel_const[kernelIndex] * inputImage[imageIndex];
            }
        }
        if (sum < 0) {
            sum = 0;
        }
        if (sum > 255) {
            sum = 255;
        }
        resultImage[channelIndex * const_channelN + i * const_resultCols + j] = sum;
    }
}


Image* processImage_threeChannel_constant_noGrid(Image* h_image, Kernel* h_kernel) {
    Image* h_resultImage = initColoredImage(h_image->rows - 2 * h_kernel->paddingSize, h_image->cols - 2 * h_kernel->paddingSize);
    long long int totalResultPixels = h_resultImage->rows * h_resultImage->cols * (long long int)3;
    
    short int* h_resultImagePixels = new short int[totalResultPixels];
    short int* h_imagePixels = linearizeImage(h_image);

    int threadsPerBlock = 1024; 
    dim3 numBlocks((h_resultImage->cols * h_resultImage->rows + threadsPerBlock - 1) / threadsPerBlock, 1, 3);

    short int* d_resultImagePixels;
    short int* d_imagePixels;

    long long int totalInputPixels = h_image->rows * h_image->cols * 3;

    cudaMalloc(&d_resultImagePixels, (size_t)totalResultPixels*sizeof(short int));
    cudaMalloc(&d_imagePixels, (size_t)totalInputPixels*sizeof(short int));

    cudaMemcpyToSymbol(d_kernel_const, h_kernel->values, h_kernel->rows * h_kernel->cols * sizeof(float));
    cudaMemcpyToSymbol(const_resultCols, &h_resultImage->cols, sizeof(int));
    cudaMemcpyToSymbol(const_resultRows, &h_resultImage->rows, sizeof(int));
    cudaMemcpyToSymbol(const_paddingSize, &h_kernel->paddingSize, sizeof(int));
    cudaMemcpyToSymbol(const_inputImageCols, &h_image->cols, sizeof(int));
    cudaMemcpyToSymbol(const_kernelCols, &h_kernel->cols, sizeof(int));
    cudaMemcpyToSymbol(const_N, &totalResultPixels, sizeof(int));
    int channelN = h_resultImage->rows * h_resultImage->cols;
    cudaMemcpyToSymbol(const_channelN, &channelN, sizeof(int));
    int channelNInput = h_image->rows * h_image->cols;
    cudaMemcpyToSymbol(const_channelNInput, &channelNInput, sizeof(int));

    cudaMemcpy(d_imagePixels, h_imagePixels, totalInputPixels*sizeof(short int), cudaMemcpyHostToDevice);

    kernelProcessing_constant_noGrid<<<numBlocks, threadsPerBlock>>>(d_imagePixels, d_resultImagePixels);
    
    cudaMemcpy(h_resultImagePixels, d_resultImagePixels, totalResultPixels*sizeof(short int), cudaMemcpyDeviceToHost);

    delinearizeImage(h_resultImagePixels, h_resultImage);

    delete[] h_resultImagePixels;
    delete[] h_imagePixels;

    cudaFree(d_imagePixels);
    cudaFree(d_resultImagePixels);

    return h_resultImage;
}


__global__ void kernelProcessing_threeChannelTogether(short int* inputImage0,
                                    short int* inputImage1, 
                                    short int* inputImage2,
                                    short int* resultImage0,
                                    short int* resultImage1,
                                    short int* resultImage2) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = index / const_resultCols;
    int j = index % const_resultCols;

    if (index < const_N){
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
        for (int k = -const_paddingSize; k <= const_paddingSize; ++k) {
            for (int l = -const_paddingSize; l <= const_paddingSize; ++l) {
                int kernelIndex = (k+const_paddingSize) * const_kernelCols + (l+const_paddingSize);
                int imageIndex = (i+const_paddingSize+k) * const_inputImageCols + (j+const_paddingSize+l);
                sum0 += d_kernel_const[kernelIndex] * inputImage0[imageIndex];
                sum1 += d_kernel_const[kernelIndex] * inputImage1[imageIndex];
                sum2 += d_kernel_const[kernelIndex] * inputImage2[imageIndex];
            }
        }
        resultImage0[i * const_resultCols + j] = min(max(sum0, 0.0f), 255.0f);
        resultImage1[i * const_resultCols + j] = min(max(sum1, 0.0f), 255.0f);
        resultImage2[i * const_resultCols + j] = min(max(sum2, 0.0f), 255.0f);
    }
}


Image* processImage_CUDA_threeChannelTogether(Image* h_image, Kernel* h_kernel) {
    Image* h_resultImage = initColoredImage(h_image->rows - 2 * h_kernel->paddingSize, h_image->cols - 2 * h_kernel->paddingSize);
    long long int pixelsPerChannel = h_resultImage->rows * h_resultImage->cols;
    long long int pixelsPerChannel_inputImage = h_image->rows * h_image->cols;

    short int* d_resultImagePixels_channel0;
    short int* d_resultImagePixels_channel1;
    short int* d_resultImagePixels_channel2;

    short int* d_imagePixels_channel0;
    short int* d_imagePixels_channel1;
    short int* d_imagePixels_channel2;

    cudaMemcpyToSymbol(d_kernel_const, h_kernel->values, h_kernel->rows * h_kernel->cols * sizeof(float));
    cudaMemcpyToSymbol(const_resultCols, &h_resultImage->cols, sizeof(int));
    cudaMemcpyToSymbol(const_resultRows, &h_resultImage->rows, sizeof(int));
    cudaMemcpyToSymbol(const_paddingSize, &h_kernel->paddingSize, sizeof(int));
    cudaMemcpyToSymbol(const_inputImageCols, &h_image->cols, sizeof(int));
    cudaMemcpyToSymbol(const_kernelCols, &h_kernel->cols, sizeof(int));
    cudaMemcpyToSymbol(const_N, &pixelsPerChannel, sizeof(int));
    int channelN = h_resultImage->rows * h_resultImage->cols;
    cudaMemcpyToSymbol(const_channelN, &channelN, sizeof(int));
    int channelNInput = h_image->rows * h_image->cols;
    cudaMemcpyToSymbol(const_channelNInput, &channelNInput, sizeof(int));

    cudaMalloc(&d_resultImagePixels_channel0, (size_t)pixelsPerChannel*sizeof(short int));
    cudaMalloc(&d_resultImagePixels_channel1, (size_t)pixelsPerChannel*sizeof(short int));
    cudaMalloc(&d_resultImagePixels_channel2, (size_t)pixelsPerChannel*sizeof(short int));

    cudaMalloc(&d_imagePixels_channel0, (size_t)pixelsPerChannel_inputImage*sizeof(short int));
    cudaMalloc(&d_imagePixels_channel1, (size_t)pixelsPerChannel_inputImage*sizeof(short int));
    cudaMalloc(&d_imagePixels_channel2, (size_t)pixelsPerChannel_inputImage*sizeof(short int));

    cudaMemcpy(d_imagePixels_channel0, h_image->channels[0], pixelsPerChannel_inputImage*sizeof(short int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imagePixels_channel1, h_image->channels[1], pixelsPerChannel_inputImage*sizeof(short int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imagePixels_channel2, h_image->channels[2], pixelsPerChannel_inputImage*sizeof(short int), cudaMemcpyHostToDevice);

    int numThreads = h_resultImage->rows * h_resultImage->cols;
    int threadsPerBlock = 1024;
    int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

    kernelProcessing_threeChannelTogether<<<numBlocks, threadsPerBlock>>>(d_imagePixels_channel0, d_imagePixels_channel1, d_imagePixels_channel2, d_resultImagePixels_channel0, d_resultImagePixels_channel1, d_resultImagePixels_channel2);
    
    cudaMemcpy(h_resultImage->channels[0], d_resultImagePixels_channel0, pixelsPerChannel*sizeof(short int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_resultImage->channels[1], d_resultImagePixels_channel1, pixelsPerChannel*sizeof(short int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_resultImage->channels[2], d_resultImagePixels_channel2, pixelsPerChannel*sizeof(short int), cudaMemcpyDeviceToHost);

    cudaFree(d_imagePixels_channel0);
    cudaFree(d_imagePixels_channel1);
    cudaFree(d_imagePixels_channel2);
    cudaFree(d_resultImagePixels_channel0);
    cudaFree(d_resultImagePixels_channel1);
    cudaFree(d_resultImagePixels_channel2);

    return h_resultImage;
}


Image* processImage_CUDA_oneChannel_constant(Image* h_image, Kernel* h_kernel) {
    Image* h_resultImage = initColoredImage(h_image->rows - 2 * h_kernel->paddingSize, h_image->cols - 2 * h_kernel->paddingSize);

    int numThreads = h_resultImage->rows * h_resultImage->cols;
    int threadsPerBlock = 1024;
    int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
    
    short int* d_resultImage;
    short int* d_image;

    cudaMalloc(&d_resultImage, (size_t)h_resultImage->rows*h_resultImage->cols*sizeof(short int));
    cudaMalloc(&d_image, (size_t)h_image->rows*h_image->cols*sizeof(short int));

    cudaMemcpyToSymbol(d_kernel_const, h_kernel->values, h_kernel->rows * h_kernel->cols * sizeof(float));
    
    for (int channelIndex = 0; channelIndex < 3; channelIndex++){
        cudaMemcpy(d_image, h_image->channels[channelIndex], h_image->rows*h_image->cols*sizeof(short int), cudaMemcpyHostToDevice);
        kernelProcessing_constant_noGrid<<<numBlocks, threadsPerBlock>>>(d_image, d_resultImage);
        cudaMemcpy(h_resultImage->channels[channelIndex], d_resultImage, h_resultImage->cols*h_resultImage->rows*sizeof(short int), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_image);
    cudaFree(d_resultImage);

    return h_resultImage;
}



void deleteColoredImage(Image* image) {
    delete[] image->channels[0];
    delete[] image->channels[1];
    delete[] image->channels[2];
    delete image;
}


cv::Mat* reconstructColoredImage(Image* image) {
    cv::Mat* outputImage = new cv::Mat(image->rows, image->cols, CV_8UC3);

    for (int i = 0; i < image->rows; ++i) {
        for (int j = 0; j < image->cols; ++j) {
            outputImage->at<cv::Vec3b>(i, j)[0] = static_cast<uchar>(image->channels[0][i * image->cols + j]);
            outputImage->at<cv::Vec3b>(i, j)[1] = static_cast<uchar>(image->channels[1][i * image->cols + j]);
            outputImage->at<cv::Vec3b>(i, j)[2] = static_cast<uchar>(image->channels[2][i * image->cols + j]);
        }
    }

    return outputImage;
}


void printImage(const Image* image, const char* imageName) {
    std::cout << imageName << " size [" << image->rows << ", " << image->cols << "]" << std::endl;
}

#endif