#ifndef PARALLEL_DYNAMIC_ALGO_NO_FIRST_TOUCH_H
#define PARALLEL_DYNAMIC_ALGO_NO_FIRST_TOUCH_H

#include "parallelDynamicAlgo.h"

class ParallelDynamicAlgoNoFirstTouch : public ParallelDynamicAlgo {
    public:
        ParallelDynamicAlgoNoFirstTouch();
    protected:
        void initData(Data* data, long long int size, float* values) override;
};

#endif