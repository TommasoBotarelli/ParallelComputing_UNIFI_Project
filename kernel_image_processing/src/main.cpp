#include <iostream>
#include <opencv2/opencv.hpp>

#include "image.h"
#include "kernel.h"


int main(int, char**){
    int kernelSize = 3;
    int paddingSize = kernelSize / 2;
    float* gaussianValues = new float[9]{0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
    float* boxBlurValues = new float[9]{0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111};
    float* identityValues = new float[9]{0, -1, 0, -1, 4, -1, 0, -1, 0};
    float* sharpenValues = new float[9]{0, -1, 0, -1, 5, -1, 0, -1, 0};

    char* imagePath = "..\\images\\big_image.jpg";

    printf("Padding size: %d\n", paddingSize);
    
    Kernel* kernel = initKernel(kernelSize, kernelSize, paddingSize, boxBlurValues);

    cv::Mat imageRGB = cv::imread(imagePath, cv::IMREAD_COLOR);

    Image* rawImage = initColoredImage(&imageRGB);

    cv::copyMakeBorder(imageRGB, imageRGB, paddingSize, paddingSize, paddingSize, paddingSize, cv::BORDER_REPLICATE);

    Image* imageWithBorder = initColoredImage(&imageRGB);

    cv::imwrite("..\\images\\borderedImage.jpg", imageRGB);

    printf("Image size: %d x %d\n", rawImage->rows, rawImage->cols);
    printf("Image with border size: %d x %d\n", imageWithBorder->rows, imageWithBorder->cols);

    Image* processedImage = processImage(imageWithBorder, kernel);

    printf("Processed image size: %d x %d\n", processedImage->rows, processedImage->cols);

    cv::Mat* outputImage = reconstructColoredImage(processedImage);

    cv::imwrite("..\\images\\processedImage.jpg", *outputImage);

    deleteColoredImage(rawImage);
    deleteColoredImage(imageWithBorder);
    deleteColoredImage(processedImage);
}

