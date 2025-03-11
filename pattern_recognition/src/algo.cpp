#include "algo.h"

#include <iostream>
#include <fstream>

void Algo::printData(Data data) {
    std::cout << "Data size: " << data.size << std::endl;
    std::cout << "Values: ";
    for (int i = 0; i < data.size; i++) {
        std::cout << data.values[i] << " ";
    }
    std::cout << std::endl;
}


Data* Algo::read(const char *filename, bool printData = false)
{
    Data* data = new Data();

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return data;
    }
    
    float size;
    file.read(reinterpret_cast<char*>(&size), sizeof(float));

    float* values = new float[(int)size];

    for (int index = 0; index < size; index++) {
        file.read(reinterpret_cast<char*>(&values[index]), sizeof(float));
    }

    this->initData(data, size, values);
    
    file.close();

    if (printData) {
        printf("Successfully read file %s\n", filename);
    }

    return data;
}

char* Algo::getName() {
    return this->name;
}