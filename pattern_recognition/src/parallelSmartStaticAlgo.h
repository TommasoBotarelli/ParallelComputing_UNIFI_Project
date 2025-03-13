#ifndef PARALLEL_SMART_STATIC_ALGO_H
#define PARALLEL_SMART_STATIC_ALGO_H 

#include "parallelStaticAlgo.h"

class ParallelSmartStaticAlgo: public ParallelStaticAlgo{
    public:
        ParallelSmartStaticAlgo();
        long long int compute(Data* __restrict entireTimeSeries, Data* __restrict timeSeriesToSearch, bool printResults) override;
};

#endif