#include "parallelStaticAlgo.h"


#ifndef PARALLEL_DYNAMIC_ALGO
#define PARALLEL_DYNAMIC_ALGO

class ParallelDynamicAlgo : public ParallelStaticAlgo {
    public:
        ParallelDynamicAlgo();
        virtual long long int compute(Data* __restrict entireTimeSeries, Data* __restrict timeSeriesToSearch, bool printResults);
    protected:
        virtual void initData(Data* data, long long int size, float* values) override;
};

#endif