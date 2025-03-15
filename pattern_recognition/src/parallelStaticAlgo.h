#include "algo.h"


#ifndef PARALLEL_STATIC_ALGO
#define PARALLEL_STATIC_ALGO

class ParallelStaticAlgo : public Algo {
    public:
        ParallelStaticAlgo();
        virtual long long int compute(Data* __restrict entireTimeSeries, Data* __restrict timeSeriesToSearch, bool printResults);
    protected:
        virtual void initData(Data* data, long long int size, float* values);
        virtual void setName(char* name);
};

#endif