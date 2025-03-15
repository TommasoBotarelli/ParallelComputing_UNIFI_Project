#include "parallelStaticAlgo.h"


#ifndef PARALLEL_STATIC_ALGO_NO_FIRST_TOUCH
#define PARALLEL_STATIC_ALGO_NO_FIRST_TOUCH

class ParallelStaticAlgoNoFirstTouch : public ParallelStaticAlgo {
    public:
        ParallelStaticAlgoNoFirstTouch();
    protected:
        void initData(Data* data, long long int size, float* values) override;
};

#endif