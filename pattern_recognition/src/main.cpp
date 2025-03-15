#include <stdio.h>
#include <algorithm>
#include <map>
#include <string>
#include <iostream>
#include <filesystem>
#include "omp.h"

#include "sequentialAlgo.h"
#include "parallelStaticAlgo.h"
#include "parallelSmartStaticAlgo.h"
#include "parallelStaticAlgoNoFirstTouch.h"
#include "parallelManualAlgo.h"
#include "parallelManualAlgoNoFirstTouch.h"
#include "parallelDynamicAlgo.h"
#include "parallelDynamicAlgoNoFirstTouch.h"


#define NUM_ITERATIONS 10


bool compStringToInt(std::string a, std::string b){
    return std::stoi(a) < std::stoi(b);
}

bool compString(std::string a, std::string b){
    return a > b;
}

void cleanData(Data* data){
    delete[] data->values;
    delete data;
}

std::map<std::string, std::vector<std::string>>* getSeriesPath(std::string path){
    std::map<std::string, std::vector<std::string>>* folders = new std::map<std::string, std::vector<std::string>>();

    std::vector<std::string>* folderPaths = new std::vector<std::string>();
    for (const auto& entry : std::filesystem::directory_iterator(path)){
        folderPaths->push_back(entry.path().filename().string());
    }
    std::sort(folderPaths->begin(), folderPaths->end(), compStringToInt);
    for(int i = 0; i < (int)(folderPaths->size()); i++){
        std::string name = folderPaths->at(i);
        folderPaths->at(i) = path + "/" + name;
    }

    for (std::string name : *folderPaths){
        std::vector<std::string>* files = new std::vector<std::string>();

        for (const auto& entry : std::filesystem::directory_iterator(name)){
            files->push_back(entry.path().filename().string());
        }
        std::sort(files->begin(), files->end(), compString);

        folders->insert(std::pair<std::string, std::vector<std::string>>(name, *files));
    }

    return folders;
}


void saveComputationTimes(std::map<std::string, std::map<std::string, std::vector<long long int>>>* computationTimeElapsed, std::string filename){
    FILE* file = fopen(filename.c_str(), "w");
    if (file == NULL){
        printf("Error opening file!\n");
        exit(1);
    }

    fprintf(file, "Algorithm;Series;File;");
    for (int i = 0; i < NUM_ITERATIONS; i++){
        fprintf(file, "Iteration %d", i);
        if (i < NUM_ITERATIONS - 1){
            fprintf(file, ";");
        }
    }
    fprintf(file, "\n");

    for (const auto& [algoName, seriesMap] : *computationTimeElapsed) {
        for (const auto& [seriesFiles, times] : seriesMap) {
            fprintf(file, "%s;%s;", algoName.c_str(), seriesFiles.c_str());
            for (size_t i = 0; i < times.size(); ++i) {
                fprintf(file, "%lld", times[i]);
                if (i < times.size() - 1){
                    fprintf(file, ";");
                }
            }
            fprintf(file, "\n");
        }
    }

    fclose(file);
}


void printBind(int bind){
    switch (bind) {
        case omp_proc_bind_false:
            printf("Proc bind: false\n");
            break;
        case omp_proc_bind_true:
            printf("Proc bind: true\n");
            break;
        case omp_proc_bind_master:
            printf("Proc bind: master\n");
            break;
        case omp_proc_bind_close:
            printf("Proc bind: close\n");
            break;
        case omp_proc_bind_spread:
            printf("Proc bind: spread\n");
            break;
        default:
            printf("Proc bind: unknown\n");
            break;
    }
}


int strongScalingComputation(int numThreads, std::string path, std::string outputFileName){
    omp_set_num_threads(numThreads);

    std::map<std::string, std::vector<std::string>>* serieFolders = getSeriesPath(path);
    std::map<std::string, std::map<std::string, std::vector<long long int>>> computationTimeElapsed;

    std::vector<Algo*> algos = {
        new SequentialAlgo(),
        new ParallelManualAlgoNoFirstTouch(),
        new ParallelManualAlgo(),
        new ParallelStaticAlgo(),
        new ParallelSmartStaticAlgo(),
        new ParallelStaticAlgoNoFirstTouch(),
        new ParallelDynamicAlgo(),
        new ParallelDynamicAlgoNoFirstTouch()
    };

    printBind(omp_get_proc_bind());

    long long int movingComputationTimeElapsed;
    std::map<std::string, std::vector<long long int>> computationTimeElapsedEntry;
    
    for (Algo* algo : algos){
        printf("\n#####################################################################\n");
        printf("######### Running %s #########\n", algo->getName());

        for (auto const& [serieName, files] : *serieFolders){
            std::string entireTimeSerieName = serieName + "/" + files[files.size() - 1];
            std::string timeSerieToSearchName = serieName + "/" + files[0];

            printf("Entire time serie: %s\n", entireTimeSerieName.c_str());
            printf("Time serie to search: %s\n", timeSerieToSearchName.c_str());

            long long int totalComputationTimeElapsed = 0;
            std::vector<long long int> computationTimes;

            for (int i = 0; i < NUM_ITERATIONS; i++){
                Data* entireTimeSeries = algo->read(entireTimeSerieName.c_str(), false);
                Data* timeSeriesTosearch = algo->readSerieToSearch(timeSerieToSearchName.c_str(), false);
    
                movingComputationTimeElapsed = algo->compute(entireTimeSeries, timeSeriesTosearch, false);
                
                computationTimes.push_back(movingComputationTimeElapsed);
                totalComputationTimeElapsed += movingComputationTimeElapsed;

                cleanData(entireTimeSeries);
                cleanData(timeSeriesTosearch);
            }

            computationTimeElapsedEntry.insert(std::pair<std::string, std::vector<long long int>>(entireTimeSerieName + " - " + timeSerieToSearchName, computationTimes));
            
            printf("Average time for iteration: %lld ms\n", totalComputationTimeElapsed / NUM_ITERATIONS);
        }
        
        computationTimeElapsed.insert(std::pair<std::string, std::map<std::string, std::vector<long long int>>>(algo->getName(), computationTimeElapsedEntry));
        computationTimeElapsedEntry.clear();

        printf("#####################################################################\n");
        printf("#####################################################################\n");
    }    

    saveComputationTimes(&computationTimeElapsed, outputFileName);

    delete(serieFolders);
    for (Algo* algo : algos){
        delete(algo);
    }

    return 0;
}

int weakScalingComputation(std::string path, std::string outputFileName){
    std::map<std::string, std::vector<std::string>>* serieFolders = getSeriesPath(path);
    std::map<std::string, std::map<std::string, std::vector<long long int>>> computationTimeElapsed;

    std::vector<Algo*> algos;

    printBind(omp_get_proc_bind());

    long long int movingComputationTimeElapsed;
    std::map<std::string, std::vector<long long int>> computationTimeElapsedEntry;

    std::vector<std::string> orderedFolders;
    for (auto const& [serieName, files] : *serieFolders){
        orderedFolders.push_back(std::filesystem::path(serieName).filename().string());
    }
    std::sort(orderedFolders.begin(), orderedFolders.end(), compStringToInt);

    int numThreads = 2;

    for (const std::string& folder : orderedFolders){
        omp_set_num_threads(numThreads);

        std::string serieName = path + "/" + folder;
        std::vector<std::string> files = serieFolders->at(serieName);

        algos = {
            new SequentialAlgo(),
            new ParallelManualAlgoNoFirstTouch(),
            new ParallelManualAlgo(),
            new ParallelStaticAlgo(),
            new ParallelSmartStaticAlgo()
        };

        for (Algo* algo : algos){
            printf("\n#####################################################################\n");
            printf("######### Running %s #########\n", algo->getName());

            std::string entireTimeSerieName = serieName + "/" + files[files.size() - 1];
            std::string timeSerieToSearchName = serieName + "/" + files[0];

            printf("Entire time serie: %s\n", entireTimeSerieName.c_str());
            printf("Time serie to search: %s\n", timeSerieToSearchName.c_str());

            long long int totalComputationTimeElapsed = 0;
            std::vector<long long int> computationTimes;

            for (int i = 0; i < NUM_ITERATIONS; i++){
                Data* entireTimeSeries = algo->read(entireTimeSerieName.c_str(), false);
                Data* timeSeriesTosearch = algo->readSerieToSearch(timeSerieToSearchName.c_str(), false);
    
                movingComputationTimeElapsed = algo->compute(entireTimeSeries, timeSeriesTosearch, false);
                
                computationTimes.push_back(movingComputationTimeElapsed);
                totalComputationTimeElapsed += movingComputationTimeElapsed;

                cleanData(entireTimeSeries);
                cleanData(timeSeriesTosearch);
            }

            computationTimeElapsedEntry.insert(std::pair<std::string, std::vector<long long int>>(entireTimeSerieName + " - " + timeSerieToSearchName, computationTimes));
            
            printf("Average time for iteration: %lld ms\n", totalComputationTimeElapsed / NUM_ITERATIONS);
            
            computationTimeElapsed.insert(std::pair<std::string, std::map<std::string, std::vector<long long int>>>(algo->getName(), computationTimeElapsedEntry));
            computationTimeElapsedEntry.clear();
    
            printf("#####################################################################\n");
            printf("#####################################################################\n");
        }    

        numThreads *= 2;

        for (Algo* algo : algos){
            delete(algo);
        }
    }
    

    saveComputationTimes(&computationTimeElapsed, outputFileName);

    delete(serieFolders);

    return 0;
}

 
int main(int argc, char* argv[])
{
    /*
        <path_to_series> - path to the folder containing the series. The code assumes to have a folder
            for each series and inside each folder, the files containing the series. The code will read the
            series in alphabetical order and take the first serie as the big serie in which we have to find
            the second serie.
        <threads_number> - number of threads to use in the parallel algorithms. If 0, the code will run the
            weak scaling computation. Weak scaling assumes that the number of threads is equal to the number
            of series folder in the path. So the code will run the algorithms for each series folder and increase
            the number of threads for each folder.
        <output_file_name> - name of the file to save the computation times.
    */
    
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <path_to_series> <threads_number> <output_file_name>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    int numThreads = std::stoi(argv[2]);
    std::string outputFileName = argv[3];

    int returnCode;

    if (numThreads == 0){
        returnCode = weakScalingComputation(path, outputFileName);
    }
    else{
        returnCode = strongScalingComputation(numThreads, path, outputFileName);
    }

    return returnCode;
}