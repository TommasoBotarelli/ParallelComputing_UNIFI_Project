#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/opencv.hpp>
#include "kernel.h"

struct Image {
    int rows;
    int cols;

    int* channels[3];
};


Image* initColoredImage(int rows, int cols) {
    Image* image = new Image();
    image->rows = rows;
    image->cols = cols;
    image->channels[0] = new int[rows * cols];
    image->channels[1] = new int[rows * cols];
    image->channels[2] = new int[rows * cols];
    return image;
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

Image* processImage(Image* inputImage, Kernel* kernel) {
    Image* processedImage = initColoredImage(inputImage->rows - 2 * kernel->paddingSize, inputImage->cols - 2 * kernel->paddingSize);

    for (int channelIndex = 0; channelIndex < 3; ++channelIndex) {
        for (int i = kernel->paddingSize; i < inputImage->rows - kernel->paddingSize; ++i) {
            for (int j = kernel->paddingSize; j < inputImage->cols - kernel->paddingSize; ++j) {
                processedImage->channels[channelIndex][(i-kernel->paddingSize) * processedImage->cols + (j-kernel->paddingSize)] = 0.0;
                for (int k = -kernel->paddingSize; k <= kernel->paddingSize; ++k) {
                    for (int l = -kernel->paddingSize; l <= kernel->paddingSize; ++l) {
                        processedImage->channels[channelIndex][(i-kernel->paddingSize) * processedImage->cols + (j-kernel->paddingSize)] += kernel->values[(k+kernel->paddingSize) * kernel->cols + (l+kernel->paddingSize)] * 
                            static_cast<int>(inputImage->channels[channelIndex][(i+k) * inputImage->cols + (j+l)]);
                    }
                }
                if (processedImage->channels[channelIndex][(i-kernel->paddingSize) * processedImage->cols + (j-kernel->paddingSize)] < 0) {
                    processedImage->channels[channelIndex][(i-kernel->paddingSize) * processedImage->cols + (j-kernel->paddingSize)] = 0;
                }
                if (processedImage->channels[channelIndex][(i-kernel->paddingSize) * processedImage->cols + (j-kernel->paddingSize)] > 255) {
                    processedImage->channels[channelIndex][(i-kernel->paddingSize) * processedImage->cols + (j-kernel->paddingSize)] = 255;
                }
            }
        }
    }

    return processedImage;
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

#endif