#include <cmath>
#include <stdio.h>
#include <chrono>
#include <string>

#include "parallelDynamicAlgo.h"
#include "omp.h"

ParallelDynamicAlgo::ParallelDynamicAlgo()
{
    parallel = true;
    vectorized = false;

    this->setName("Dynamic parallel implementation");
}


long long int ParallelDynamicAlgo::compute(Data* __restrict entireTimeSeries, Data* __restrict timeSeriesToSearch, bool printResults) {
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

    #pragma omp parallel for private(distance) shared(minDistance) schedule(dynamic)
    for (int i = 0; i < n - m; i++) {
        distance = 0;
        #pragma omp simd
        for (int j = 0; j < m; j++) {
            distance += abs(entireTimeSeriesValues[i + j] - timeSeriesToSearchValues[j]);
        }
        #pragma omp critical
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

void ParallelDynamicAlgo::initData(Data *data, int size, float *values)
{
    data->size = size;
    data->values = new float[size];
    #pragma omp parallel for simd schedule(dynamic)
    for(int i = 0; i < size; i++) {
        data->values[i] = values[i];
    }
}
