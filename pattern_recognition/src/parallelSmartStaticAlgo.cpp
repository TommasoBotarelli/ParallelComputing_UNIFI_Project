#include "parallelSmartStaticAlgo.h"

#include <cmath>
#include <stdio.h>
#include <chrono>
#include <string>

ParallelSmartStaticAlgo::ParallelSmartStaticAlgo()
{
    this->setName("Smart static parallel implementation");

    this->parallel = true;
    this->vectorized = true;
}

long long int ParallelSmartStaticAlgo::compute(Data *__restrict entireTimeSeries, Data *__restrict timeSeriesToSearch, bool printResults)
{
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

    #pragma omp parallel for private(distance) reduction(min:minDistance) schedule(static)
    for (int i = 0; i < n - m; i++) {
        distance = 0;
        #pragma omp simd
        for (int j = 0; j < m; j++) {
            distance += abs(entireTimeSeriesValues[i + j] - timeSeriesToSearchValues[j]);
        }

        if (distance < minDistance) {
            #pragma omp critical
            {
                if (distance < minDistance) {
                    minDistance = distance;
                    minIndex = i;
                    maxIndex = i + m;
                }
            }
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