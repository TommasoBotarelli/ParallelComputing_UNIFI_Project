#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <iomanip>

#include "image.h"
#include "kernel.h"
#include <string>

#define CUDA_DEVICE 1

double processRoutine(std::function<Image*(Image*, Kernel*)> function, Image* imageWithBorder, Kernel* kernel, int numIterations, std::string outputPath, bool print) {
    std::chrono::duration<double, std::micro> duration;
    long double totalDuration = 0.0;
    Image* processedImage;

    for (int i = 0; i < numIterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        processedImage = function(imageWithBorder, kernel);
        auto end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        if (print){
            std::cout << "[" << i << "] "<< duration.count() << " microseconds" << std::endl;
        }
        totalDuration += (std::chrono::duration_cast<std::chrono::milliseconds>(duration)).count();
        if (i < numIterations - 1){
            deleteColoredImage(processedImage);
        }
    }

    printImage(processedImage, "Processed Image");

    //cv::Mat* outputImage = reconstructColoredImage(processedImage);

    //std::cout << "Saving " << outputPath << std::endl;
    //cv::imwrite(outputPath, *outputImage);
    deleteColoredImage(processedImage);

    return totalDuration / numIterations;
}

int process(Image* imageWithBorder, int kernelSize, int kernelType, char* imagePath, int numIterations, double sequentialAverage = 0, bool print = false) {
    Kernel* kernel = getKernel(kernelSize, kernelType);
    printKernel(kernel);

    std::string outputPath;

    double averageTimeSequential = sequentialAverage;
    if (sequentialAverage == 0) {
        // Standard processing v1
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "SEQUENTIAL processing..." << std::endl;

        outputPath = std::string(imagePath) + "processedImage_sequential.jpg";
        averageTimeSequential = processRoutine(&processImage, imageWithBorder, kernel, numIterations, outputPath, print);
    }

    // CUDA setup
    cudaError_t err = cudaSuccess;
    err = cudaSetDevice(CUDA_DEVICE);

    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    
    // CUDA processing v1
    // one-channel parallelization -> no use of linearization
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "[CUDA V1] one channel (no linearization) no constant processing..." << std::endl;

    outputPath = std::string(imagePath) + "processedImage_CUDA_oneChannel_noConstant.jpg";
    double averageTimeCUDAv1 = processRoutine(&processImage_CUDA_oneChannel, imageWithBorder, kernel, numIterations, outputPath, print);

    // CUDA processing v3
    // same as v2 but I use __constant__ variable for kernel and all the other settings
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "[CUDA V2] three channel (blocks grid) with constant processing..." << std::endl;

    outputPath = std::string(imagePath) + "processedImage_CUDA_threeChannel_constant.jpg";
    double averageTimeCUDAv2 = processRoutine(&processImage_CUDA_threeChannel_constant, imageWithBorder, kernel, numIterations, outputPath, print);
    

    // CUDA processing v4
    // the pixels are linearized into a big array
    // then, instead of creating a grid of blocks I use only a dimension like the first version
    // this allows to optimize cache utilization between one single block.
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "[CUDA V3] three channel (no blocks grid) with constant processing..." << std::endl;

    outputPath = std::string(imagePath) + "processedImage_threeChannel_constant_noGrid.jpg";
    double averageTimeCUDAv3 = processRoutine(&processImage_threeChannel_constant_noGrid, imageWithBorder, kernel, numIterations, outputPath, print);
   
    // CUDA processing v5
    // I pass all the channel one by one at the beginning then I use a kernel modified
    // to have as input all the channel divided
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "[CUDA V4] three channel all together processing..." << std::endl;

    outputPath = std::string(imagePath) + "processedImage_CUDA_threeChannelTogether.jpg";
    double averageTimeCUDAv4 = processRoutine(&processImage_CUDA_threeChannelTogether, imageWithBorder, kernel, numIterations, outputPath, print);
    
    // CUDA processing v6
    // same as v1 but with the use of __constant__
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "[CUDA V5] one channel with constant (no linearization) processing..." << std::endl;

    outputPath = std::string(imagePath) + "processedImage_CUDA_oneChannel_constant.jpg";
    double averageTimeCUDAv5 = processRoutine(&processImage_CUDA_oneChannel_constant, imageWithBorder, kernel, numIterations, outputPath, print);

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    std::cout << "AVERAGE TIMES for " << numIterations << " iterations" << std::endl;
    
    std::cout << std::setw(30) << std::left << "Seq: " << std::setw(10) << std::right << averageTimeSequential << std::endl;
    
    std::cout << std::setw(30) << std::left << "1channel_{noConst}: " << std::setw(10) << std::right << averageTimeCUDAv1 << std::endl;
    std::cout << std::setw(30) << std::left << "3channel_{grid}: " << std::setw(10) << std::right << averageTimeCUDAv2 << std::endl;
    std::cout << std::setw(30) << std::left << "3channel: " << std::setw(10) << std::right << averageTimeCUDAv3 << std::endl;
    std::cout << std::setw(30) << std::left << "3channel_{3}: " << std::setw(10) << std::right << averageTimeCUDAv4 << std::endl;
    std::cout << std::setw(30) << std::left << "1channel: " << std::setw(10) << std::right << averageTimeCUDAv5 << std::endl;
    
    double speedUpv1 = averageTimeSequential / averageTimeCUDAv1;
    double speedUpv3 = averageTimeSequential / averageTimeCUDAv2;
    double speedUpv4 = averageTimeSequential / averageTimeCUDAv3;
    double speedUpv5 = averageTimeSequential / averageTimeCUDAv4;
    double speedUpv6 = averageTimeSequential / averageTimeCUDAv5;
    std::cout << std::setw(30) << std::left << "Speed up 1channel_{noConst}: " << std::setw(10) << std::right << speedUpv1 << std::endl;
    std::cout << std::setw(30) << std::left << "Speed up 3channel_{grid}: " << std::setw(10) << std::right << speedUpv3 << std::endl;
    std::cout << std::setw(30) << std::left << "Speed up 3channel: " << std::setw(10) << std::right << speedUpv4 << std::endl;
    std::cout << std::setw(30) << std::left << "Speed up 3channel_{3}: " << std::setw(10) << std::right << speedUpv5 << std::endl;
    std::cout << std::setw(30) << std::left << "Speed up 1channel: " << std::setw(10) << std::right << speedUpv6 << std::endl;
    
    deleteColoredImage(imageWithBorder);
    
    return 0;
}


int testMode(int imageWidth, int imageHeight, int kernelSize, int kernelType, char* imagePath, int numIterations, int numRowsTest, int numColsTest, double averageTimeSequential, bool print) {
    // get test image
    Image* rawImage = getTestImage(imageWidth, imageHeight, numRowsTest, numColsTest);
    
    printImage(rawImage, "Initial Image");
    cv::Mat* imageRGB = reconstructColoredImage(rawImage);
    
    //std::string outputPath = std::string(imagePath) + "initialImage.jpg";
    //std::cout << "Saving " << outputPath << std::endl;
    //cv::imwrite(outputPath, *imageRGB);
    
    // get image with border
    int paddingSize = kernelSize / 2;
    Image* imageWithBorder = getBorderedImage(rawImage, paddingSize);
    printImage(imageWithBorder, "Image with Border");
    cv::Mat* imageWithBorderRGB = reconstructColoredImage(imageWithBorder);

    //outputPath = std::string(imagePath) + "imageWithBorder.jpg";
    //std::cout << "Saving " << outputPath << std::endl;
    //cv::imwrite(outputPath, *imageWithBorderRGB);

    // process
    int returnCode = process(imageWithBorder, kernelSize, kernelType, imagePath, numIterations, averageTimeSequential, print);

    deleteColoredImage(rawImage);

    return returnCode;
}

int imageMode(char* inputImageFilePath, int kernelSize, int kernelType, char* imagePath, int numIterations, double averageTimeSequential, bool print) {
    // read the image
    cv::Mat imageRGB = cv::imread(inputImageFilePath, cv::IMREAD_COLOR);
    Image* rawImage = initColoredImage(&imageRGB);

    std::string printImageString = "Initial image (" + std::string(inputImageFilePath) + ")"; 
    printImage(rawImage, printImageString.c_str());
    
    // construct the image with border
    int paddingSize = kernelSize / 2;
    cv::copyMakeBorder(imageRGB, imageRGB, paddingSize, paddingSize, paddingSize, paddingSize, cv::BORDER_REPLICATE);
    Image* imageWithBorder = initColoredImage(&imageRGB);
    printImage(imageWithBorder, "Image with Border");
    
    std::string outputPath = std::string(imagePath) + "imageWithBorder.jpg";
    std::cout << "Saving " << outputPath << std::endl;
    cv::imwrite(outputPath, imageRGB);

    // process
    int returnCode = process(imageWithBorder, kernelSize, kernelType, imagePath, numIterations, averageTimeSequential, print);

    deleteColoredImage(rawImage);

    return returnCode;
}


int main(int argc, char** argv){
    cudaDeviceProp deviceProp;
    int device;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);

    std::cout << "Device: " << deviceProp.name << std::endl;
    std::cout << "Total constant memory: " << deviceProp.totalConstMem / 1024 << " KB" << std::endl;

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <image_file_path> <kernel_size> <kernel_type> <saving_image_path> <num_iterations> <average_time_sequential_in>" << std::endl;
        std::cerr << "Usage: " << argv[0] << " test <image_width> <image_height> <kernel_size> <kernel_type> <saving_image_path> <num_iterations> <num_test_rows> <num_test_cols> <average_time_sequential_in>" << std::endl;
        return 1;
    }

    if (strcmp("test", argv[1]) == 0){
        int imageWidth = std::stoi(argv[2]);
        int imageHeight = std::stoi(argv[3]);
        int kernelSize = std::stoi(argv[4]);
        int kernelType = std::stoi(argv[5]);
        char* imagePath = argv[6];
        int numIterations = std::stoi(argv[7]);
        int numRowsTest = std::stoi(argv[8]);
        int numColsTest = std::stoi(argv[9]);
        double averageTimeSequential = (double)std::stoi(argv[10]);
        bool print = std::stoi(argv[11]);

        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Recap of settings:" << std::endl;
        std::cout << "Image Width: " << imageWidth << std::endl;
        std::cout << "Image Height: " << imageHeight << std::endl;
        std::cout << "Kernel Size: " << kernelSize << std::endl;
        std::cout << "Kernel Type: " << getKernelTypeString(kernelType) << std::endl;
        std::cout << "Saving Image Path: " << imagePath << std::endl;
        std::cout << "Number of Iterations: " << numIterations << std::endl;
        std::cout << "Number of Test Rows: " << numRowsTest << std::endl;
        std::cout << "Number of Test Columns: " << numColsTest << std::endl;
        std::cout << "Print: " << print << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;

        testMode(imageWidth, imageHeight, kernelSize, kernelType, imagePath, numIterations, numRowsTest, numColsTest, averageTimeSequential, print);
    }
    else {
        char* imageFilePath = argv[1];
        int kernelSize = std::stoi(argv[2]);
        int kernelType = std::stoi(argv[3]);
        char* savingImagePath = argv[4];
        int numIterations = std::stoi(argv[5]);
        double averageTimeSequential = (double)std::stoi(argv[6]);
        bool print = std::stoi(argv[7]);

        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Recap of settings:" << std::endl;
        std::cout << "Image File Path: " << imageFilePath << std::endl;
        std::cout << "Kernel Size: " << kernelSize << std::endl;
        std::cout << "Kernel Type: " << getKernelTypeString(kernelType) << std::endl;
        std::cout << "Saving Image Path: " << savingImagePath << std::endl;
        std::cout << "Number of Iterations: " << numIterations << std::endl;
        std::cout << "Print: " << print << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;

        imageMode(imageFilePath, kernelSize, kernelType, savingImagePath, numIterations, averageTimeSequential, print);
    }

    return 0;
}

