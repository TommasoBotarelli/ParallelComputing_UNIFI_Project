#ifndef PARALLEL_MANUAL_ALGO_NO_FIRST_TOUCH_H
#define PARALLEL_MANUAL_ALGO_NO_FIRST_TOUCH_H

#include "parallelManualAlgo.h"

class ParallelManualAlgoNoFirstTouch: public ParallelManualAlgo {
    public:
        ParallelManualAlgoNoFirstTouch();
    protected:
        void initData(Data* data, long long int size, float* values) override;
};

#endif