#include "parallelStaticAlgo.h"

#ifndef PARALLEL_MANUAL_ALGO
#define PARALLEL_MANUAL_ALGO

class ParallelManualAlgo : public ParallelStaticAlgo {
    public:
        ParallelManualAlgo();
        virtual long long int compute(Data* __restrict entireTimeSeries, Data* __restrict timeSeriesToSearch, bool printResults);
    protected:
        void initData(Data* data, int size, float* values);
};

#endif