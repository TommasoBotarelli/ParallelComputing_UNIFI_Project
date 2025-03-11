#include <cmath>
#include <stdio.h>
#include <chrono>
#include <string>

#include "parallelManualAlgo.h"
#include "omp.h"


ParallelManualAlgo::ParallelManualAlgo() {
    parallel = true;
    vectorized = false;

    this->setName("Manual parallel implementation");
}


void ParallelManualAlgo::initData(Data* data, int size, float* values) {
    data->size = size;
    data->values = new float[size];
    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        int numThreads = omp_get_num_threads();

        int startIndex = size * threadId / numThreads;
        int endIndex = size * (threadId + 1) / numThreads;

        #pragma omp simd
        for (int i = startIndex; i < endIndex; i++) {
            data->values[i] = values[i];
        }
    }
}


long long int ParallelManualAlgo::compute(Data* __restrict entireTimeSeries, Data* __restrict timeSeriesToSearch, bool printResults) {
    int n = entireTimeSeries->size;
    int m = timeSeriesToSearch->size;

    if (printResults) {
        printf("Entire time series size: %d\n", n);
        printf("Time series to search size: %d\n", m);
    }

    float globalMinDistance = INFINITY;
    int globalMinIndex = -1;
    int globalThreadNumberOfFounder = -1;
    
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel shared(globalMinDistance, globalMinIndex, globalThreadNumberOfFounder)
    {
        int threadId = omp_get_thread_num();
        int numThreads = omp_get_num_threads();

        int startIndex = n * threadId / numThreads;
        int endIndex = n * (threadId + 1) / numThreads;

        if (endIndex == n){
            endIndex -= m;
        }

        if (printResults) {
            printf("Thread %d: [%d, %d]\n", threadId, startIndex, endIndex);
        }

        float* entireTimeSeriesValues = entireTimeSeries->values;
        float* timeSeriesToSearchValues = timeSeriesToSearch->values;

        float distance;
        float minDistance = INFINITY;
        int minIndex = -1;
        int i, j;

        for (i = startIndex; i < endIndex; i++) {
            distance = 0;
            #pragma omp simd
            for (j = 0; j < m; j++) {
                distance += abs(entireTimeSeriesValues[i + j] - timeSeriesToSearchValues[j]);
            }
            if (distance < minDistance) {
                minDistance = distance;
                minIndex = i;
            }
        }

        #pragma omp critical
        {
            if (minDistance < globalMinDistance) {
                globalMinDistance = minDistance;
                globalMinIndex = minIndex;
                globalThreadNumberOfFounder = threadId;
                if (printResults) {
                    printf("Thread %d found a new minimum distance: %f\n", threadId, minDistance);
                }
            }
        }
    }

    int globalMaxIndex = globalMinIndex + m;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (printResults){
        printf("---------------------------------\n");
        printf("Minimum distance: %f\n", globalMinDistance);
        printf("Interval match: [%d, %d]\n", globalMinIndex, globalMaxIndex);
        printf("Time elapsed: %lld ms\n", duration.count());
        printf("Found by thread %d\n", globalThreadNumberOfFounder);
        printf("--------------------------------\n");
    }

    return duration.count();
}