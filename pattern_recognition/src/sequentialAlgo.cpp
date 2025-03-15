#include "sequentialAlgo.h"

#include <cmath>
#include <stdio.h>
#include <chrono>


SequentialAlgo::SequentialAlgo() {
    this->parallel = false;
    this->vectorized = false;
    this->name = "Sequential implementation";
}


void SequentialAlgo::initData(Data* data, long long int size, float* values) {
    data->size = size;
    data->values = new float[size];
    for(int i = 0; i < size; i++) {
        data->values[i] = values[i];
    }
}


long long int SequentialAlgo::compute(Data* __restrict entireTimeSeries, Data* __restrict timeSeriesToSearch, bool printResults) {
    int i, j;

    int n = entireTimeSeries->size;
    int m = timeSeriesToSearch->size;

    if (printResults) {
        printf("Entire time series size: %d\n", n);
        printf("Time series to search size: %d\n", m);
    }

    float* entireTimeSeriesValues = entireTimeSeries->values;
    float* timeSeriesToSearchValues = timeSeriesToSearch->values;
    
    float minDistance = INFINITY;
    int minIndex = -1;
    int maxIndex = -1;
    float distance = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (i = 0; i < n - m; i++) {
        distance = 0;
        #pragma omp simd
        for (j = 0; j < m; j++) {
            distance += abs(entireTimeSeriesValues[i + j] - timeSeriesToSearchValues[j]);
        }

        if (distance < minDistance) {
            minDistance = distance;
            minIndex = i;
            maxIndex = i + m;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (printResults){
        printf("Minimum distance: %f\n", minDistance);
        printf("Interval match: [%d, %d]\n", minIndex, maxIndex);
        printf("Time elapsed: %lld ms\n", duration.count());
    }

    return duration.count();
}