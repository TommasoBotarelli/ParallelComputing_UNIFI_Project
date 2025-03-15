#include "algo.h"

#ifndef SEQUENTIAL_ALGO
#define SEQUENTIAL_ALGO

class SequentialAlgo : public Algo {
    public:
        SequentialAlgo();
        long long int compute(Data* __restrict entireTimeSeries, Data* __restrict timeSeriesToSearch, bool printResults);
    protected:
        void initData(Data* data, long long int size, float* values);
};

#endif