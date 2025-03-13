#include "parallelStaticAlgoNoFirstTouch.h"

ParallelStaticAlgoNoFirstTouch::ParallelStaticAlgoNoFirstTouch()
{
    this->setName("Parallel Static Algo No First Touch");

    this->parallel = true;
    this->vectorized= true;
}

void ParallelStaticAlgoNoFirstTouch::initData(Data *data, int size, float *values)
{
    data->size = size;
    data->values = new float[size];
    for(int i = 0; i < size; i++) {
        data->values[i] = values[i];
    }
}
