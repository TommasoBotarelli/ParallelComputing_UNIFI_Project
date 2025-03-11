#include <vector>

#ifndef ALGO 
#define ALGO

struct Data {
    int size;
    float* values;
};

class Algo {
    public:
        Algo(){};
        virtual ~Algo(){};
        Data* read(const char* filename, bool printData);
        void printData(Data data);

        virtual char* getName();
        
        virtual long long int compute(Data* __restrict entireTimeSeries, Data* __restrict timeSeriesToSearch, bool printResults) = 0;

    protected:
        bool parallel;
        bool vectorized;
        char* name;

        virtual void initData(Data* data, int size, float* values) = 0;
};

#endif